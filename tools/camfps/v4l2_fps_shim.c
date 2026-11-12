/* LD_PRELOAD shim for v4l2_camera 0.6.2, which never sets a frame rate.
 * After every successful VIDIOC_S_FMT (which makes the UVC driver reset the
 * interval to its 30 fps default), request CAM_FPS frames/s via VIDIOC_S_PARM.
 * Images are untouched: same node, same format, same cv_bridge colour conversion.
 * Usage: CAM_FPS=10 LD_PRELOAD=~/camfps/v4l2_fps_shim.so ros2 run v4l2_camera ...   */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <linux/videodev2.h>

typedef int (*ioctl_fn)(int, unsigned long, ...);

int ioctl(int fd, unsigned long req, ...) {
    static ioctl_fn real = NULL;
    if (!real) real = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    int r = real(fd, req, arg);
    if (r == 0 && req == VIDIOC_S_FMT) {
        const char *e = getenv("CAM_FPS");
        int fps = e ? atoi(e) : 10;
        if (fps > 0) {
            struct v4l2_streamparm p; memset(&p, 0, sizeof p);
            p.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            p.parm.capture.timeperframe.numerator = 1;
            p.parm.capture.timeperframe.denominator = fps;
            int s = real(fd, VIDIOC_S_PARM, &p);
            fprintf(stderr, "[v4l2_fps_shim] requested %d fps -> %s, now %u/%u s per frame\n", fps,
                    s == 0 ? "ok" : "FAILED", p.parm.capture.timeperframe.numerator,
                    p.parm.capture.timeperframe.denominator);
        }
    }
    return r;
}
