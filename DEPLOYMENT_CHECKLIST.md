# BTXRD Deployment Checklist

## On your PC before going to Jetson

- [ ] Copy `.env.local` to `.env` and fill in YOUR paths (see comments in `.env.local`)
- [ ] Run: `python export_onnx.py` — confirm both models show PASSED
- [ ] Copy `classification_student.onnx` and `segmentation_student.onnx` to USB stick under `/models/`
- [ ] Copy `gemma-2-2b-it-Q4_K_M.gguf` to USB stick under `/models/`
- [ ] Copy entire project folder to USB stick

## On the Jetson

- [ ] Plug in USB stick, run: `sudo mkdir -p /data && sudo mount /dev/sda1 /data`
- [ ] Add to `/etc/fstab`: `/dev/sda1  /data  exfat  defaults,noatime  0  2`
- [ ] Verify models: `ls /data/models/` — should show the 3 files above
- [ ] Install Docker: `sudo apt install docker.io`
- [ ] Install nvidia-container-runtime (see NVIDIA docs for JetPack version)
- [ ] cd into project folder
- [ ] Run: `docker build -t btxrd-backend .` (takes 20-40 min first time)
- [ ] Convert ONNX to TRT:
      ```
      trtexec --onnx=/data/models/classification_student.onnx --saveEngine=/data/models/classify.trt --fp16
      trtexec --onnx=/data/models/segmentation_student.onnx  --saveEngine=/data/models/segment.trt  --fp16
      ```
- [ ] Update `.env.jetson`: `CLASSIFY_MODEL_FILE=classify.trt`, `SEGMENT_MODEL_FILE=segment.trt`
- [ ] Run: `docker compose up`
- [ ] Open `http://JETSON_IP:3000` from another device on same network
- [ ] Upload a test X-ray — confirm classify + segment + report all work
- [ ] Set up autostart: `sudo systemctl enable btxrd`
