savedcmd_/home/hp/gpgpu-driver/gpgpu_drv.mod := printf '%s\n'   gpgpu_drv.o | awk '!x[$$0]++ { print("/home/hp/gpgpu-driver/"$$0) }' > /home/hp/gpgpu-driver/gpgpu_drv.mod
