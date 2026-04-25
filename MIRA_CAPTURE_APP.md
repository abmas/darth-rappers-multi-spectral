# Mira 220 Mac Capture App

This is a first desktop-control version for the Raspberry Pi Mira 220 setup.
It runs on the Mac, starts a live MJPEG preview on the Pi over SSH, shows that
preview locally, and captures a raw frame with the same `libcamera-raw` flow as
`capture-mira-codex.py`.

## Install Mac dependencies

```bash
python3 -m pip install -r macos_requirements.txt
```

The Mac app needs `paramiko` for SSH and `Pillow` for showing JPEG preview
frames in Tkinter.

## Raspberry Pi requirement

The preview server uses `picamera2`, which is normally available on Raspberry Pi
OS camera builds:

```bash
python3 -c "import picamera2"
```

If that import fails on the Pi, install the Raspberry Pi camera packages there
before using the preview app.

## Run

```bash
python3 mira_capture_app.py --host 192.168.1.239 --user pi --password pi
```

Click `Start Preview` to connect to the Pi and view the camera. Click
`Capture Raw` to stop the preview briefly, run the raw capture, download the
file, convert it, and restart the preview.

Captures are saved under:

```text
captures/YYYYMMDD_HHMMSS/
```

Each capture folder contains:

```text
image.raw
image12in16.raw
image_scaled_to_16bit.raw
```

## If preview does not start

The app uploads and runs `/tmp/mira_preview_server.py` on the Pi. To inspect the
Pi-side preview error:

```bash
ssh pi@192.168.1.239 'cat /tmp/mira_preview.log'
```
