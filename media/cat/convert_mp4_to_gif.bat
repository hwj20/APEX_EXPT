@echo off
setlocal enabledelayedexpansion

for %%f in (*.mp4 *.mov *.avi) do (
    echo processing：%%f
    ffmpeg -i "%%f" -vf "fps=10,scale=600:-1:flags=lanczos" "%%~nf.gif"
)
echo OK
pause