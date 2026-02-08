import argparse
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

import forensic_mark

SECONDS_PER_MINUTE = 60


def parse_start_time(value):
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_model(model_path):
    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

    signature = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_secret = sess.graph.get_tensor_by_name(signature.inputs['secret'].name)
    input_image = sess.graph.get_tensor_by_name(signature.inputs['image'].name)
    output_stegastamp = sess.graph.get_tensor_by_name(signature.outputs['stegastamp'].name)
    return sess, input_secret, input_image, output_stegastamp


def resize_for_model(frame_rgb, size):
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    pil_image = Image.fromarray(frame_rgb)
    fitted = ImageOps.fit(pil_image, size, method=resample)
    return np.array(fitted, dtype=np.float32) / 255.0


def resize_for_output(stegastamp, size):
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    stega_img = Image.fromarray(stegastamp)
    resized = stega_img.resize(size, resample=resample)
    return np.array(resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--save_video', type=str, required=True)
    parser.add_argument('--location_id', type=int, required=True)
    parser.add_argument('--location_bits', type=int, choices=[19, 20], default=20)
    parser.add_argument('--adapter_bits', type=str, default=None)
    parser.add_argument('--start_time', type=str, default=None,
                        help="ISO-8601 start time (defaults to now in UTC)")
    parser.add_argument('--fourcc', type=str, default="XVID",
                        help="FourCC codec for output video (default: XVID)")
    parser.add_argument('--segment_minutes', type=int, default=5)
    parser.add_argument('--frames_per_payload', type=int, default=None,
                        help="Override frames per payload window (default: segment length)")
    args = parser.parse_args()

    start_time = parse_start_time(args.start_time)
    sess, input_secret, input_image, output_stegastamp = load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError("Unable to open input video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if len(args.fourcc) != 4:
        raise ValueError("FourCC codec must be four characters.")
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
    if not out.isOpened():
        raise ValueError("Unable to open output video writer.")

    frames_per_payload = args.frames_per_payload
    if frames_per_payload is None:
        frames_per_payload = max(1, int(round(fps * args.segment_minutes * SECONDS_PER_MINUTE)))
    if frames_per_payload <= 0:
        raise ValueError("Frames per payload must be positive.")

    frame_index = 0
    current_secret = None
    last_timer_index = None
    last_payload_frame = -frames_per_payload

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = timedelta(seconds=frame_index / fps)
        frame_time = start_time + elapsed
        timer_index = forensic_mark.timer_index_for_datetime(frame_time)

        if timer_index != last_timer_index or frame_index - last_payload_frame >= frames_per_payload:
            current_secret = forensic_mark.build_secret_bits(
                timer_index=timer_index,
                location_id=args.location_id,
                location_bits=args.location_bits,
                adapter_bits=args.adapter_bits,
            )
            last_timer_index = timer_index
            last_payload_frame = frame_index

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_input = resize_for_model(frame_rgb, (400, 400))
        feed_dict = {input_secret: [current_secret], input_image: [model_input]}
        hidden_img = sess.run(output_stegastamp, feed_dict=feed_dict)[0]

        rescaled = (hidden_img * 255).astype(np.uint8)
        output_rgb = resize_for_output(rescaled, (width, height))
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        out.write(output_bgr)

        frame_index += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
