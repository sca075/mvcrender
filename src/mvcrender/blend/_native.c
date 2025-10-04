#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <numpy/arrayobject.h>

/* ---------- helpers ---------- */
static inline unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* straight alpha OVER: out = fg*a + bg*(1-a); outA = fa + bgA*(1-a) */
static inline void over_rgba(
    unsigned char *dst,  /* bg RGBA */
    unsigned char fr, unsigned char fg, unsigned char fb, unsigned char fa
){
    if (fa == 255) { dst[0]=fr; dst[1]=fg; dst[2]=fb; dst[3]=255; return; }
    if (fa == 0)   { return; }

    const float a  = (float)fa / 255.0f;
    const float ia = 1.0f - a;

    const float br = (float)dst[0];
    const float bg = (float)dst[1];
    const float bb = (float)dst[2];
    const float ba = (float)dst[3];

    const int orr = (int)lroundf(fr * a + br * ia);
    const int org = (int)lroundf(fg * a + bg * ia);
    const int orb = (int)lroundf(fb * a + bb * ia);
    const int oaa = (int)lroundf((float)fa + ba * ia);

    dst[0] = clamp_u8(orr);
    dst[1] = clamp_u8(org);
    dst[2] = clamp_u8(orb);
    dst[3] = clamp_u8(oaa);
}

static int parse_rgba(PyObject *tuple, unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 4) return -1;
    long rr = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 0));
    long gg = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 1));
    long bb = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 2));
    long aa = PyLong_AsLong(PyTuple_GET_ITEM(tuple, 3));
    if (PyErr_Occurred()) return -1;
    *r = clamp_u8((int)rr); *g = clamp_u8((int)gg); *b = clamp_u8((int)bb); *a = clamp_u8((int)aa);
    return 0;
}

/* sample bg at (x,y) and compute fg OVER bg into outRGBA ints */
static inline void inline_blend_at(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int x, int y,
    unsigned char fr, unsigned char fg, unsigned char fb, unsigned char fa,
    int *or_, int *og_, int *ob_, int *oa_
){
    if (x < 0 || x >= W || y < 0 || y >= H) {
        *or_ = fr; *og_ = fg; *ob_ = fb; *oa_ = fa;
        return;
    }
    unsigned char *p = base + (npy_intp)y*sy + (npy_intp)x*sx;

    if (fa == 255) { *or_=fr; *og_=fg; *ob_=fb; *oa_=255; return; }
    if (fa == 0)   { *or_=p[0]; *og_=p[1]; *ob_=p[2]; *oa_=p[3]; return; }

    const float a  = (float)fa / 255.0f;
    const float ia = 1.0f - a;

    const float br = (float)p[0];
    const float bg = (float)p[1];
    const float bb = (float)p[2];
    const float ba = (float)p[3];

    *or_ = (int)lroundf(fr * a + br * ia);
    *og_ = (int)lroundf(fg * a + bg * ia);
    *ob_ = (int)lroundf(fb * a + bb * ia);
    *oa_ = (int)lroundf((float)fa + ba * ia);
}

/* ---------- blend_mask_inplace(image, mask, color) ---------- */
static PyObject *py_blend_mask_inplace(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *msk_obj=NULL, *color_obj=NULL;
    if (!PyArg_ParseTuple(args, "OOO", &img_obj, &msk_obj, &color_obj)) return NULL;

    unsigned char fr, fg, fb, fa;
    if (parse_rgba(color_obj, &fr, &fg, &fb, &fa) != 0) {
        PyErr_SetString(PyExc_TypeError, "foreground_color must be RGBA 4-tuple");
        return NULL;
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img) return NULL;
    PyArrayObject *msk = (PyArrayObject*)PyArray_FROM_OTF(msk_obj, NPY_BOOL, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!msk) { Py_DECREF(img); return NULL; }

    if (PyArray_NDIM(img)!=3 || PyArray_DIM(img,2)!=4 || PyArray_TYPE(img)!=NPY_UBYTE) {
        Py_DECREF(img); Py_DECREF(msk);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }
    if (PyArray_NDIM(msk)!=2 ||
        PyArray_DIM(msk,0)!=PyArray_DIM(img,0) ||
        PyArray_DIM(msk,1)!=PyArray_DIM(img,1)) {
        Py_DECREF(img); Py_DECREF(msk);
        PyErr_SetString(PyExc_ValueError, "mask must be HxW boolean matching image");
        return NULL;
    }

    const npy_intp H = PyArray_DIM(img,0);
    const npy_intp W = PyArray_DIM(img,1);
    unsigned char *img_base = (unsigned char*)PyArray_DATA(img);
    npy_bool *msk_base = (npy_bool*)PyArray_DATA(msk);

    const npy_intp sy = PyArray_STRIDE(img,0);
    const npy_intp sx = PyArray_STRIDE(img,1);
    const npy_intp smy = PyArray_STRIDE(msk,0);
    const npy_intp smx = PyArray_STRIDE(msk,1);

    if (fa == 0) { Py_DECREF(img); Py_DECREF(msk); Py_RETURN_NONE; }

    if (fa == 255) {
        Py_BEGIN_ALLOW_THREADS
        for (npy_intp y=0; y<H; ++y) {
            unsigned char *prow = img_base + y*sy;
            npy_bool *pm = (npy_bool*)((char*)msk_base + y*smy);
            for (npy_intp x=0; x<W; ++x) {
                if (*pm) {
                    unsigned char *px = prow + x*sx;
                    px[0]=fr; px[1]=fg; px[2]=fb; px[3]=255;
                }
                pm = (npy_bool*)((char*)pm + smx);
            }
        }
        Py_END_ALLOW_THREADS
        Py_DECREF(img); Py_DECREF(msk);
        Py_RETURN_NONE;
    }

    const float a  = (float)fa / 255.0f;
    const float ia = 1.0f - a;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp y=0; y<H; ++y) {
        unsigned char *prow = img_base + y*sy;
        npy_bool *pm = (npy_bool*)((char*)msk_base + y*smy);
        for (npy_intp x=0; x<W; ++x) {
            if (*pm) {
                unsigned char *px = prow + x*sx;

                const float br = (float)px[0];
                const float bg = (float)px[1];
                const float bb = (float)px[2];
                const float ba = (float)px[3];

                const int orr = (int)lroundf(fr * a + br * ia);
                const int org = (int)lroundf(fg * a + bg * ia);
                const int orb = (int)lroundf(fb * a + bb * ia);
                const int oaa = (int)lroundf((float)fa + ba * ia);

                px[0] = clamp_u8(orr);
                px[1] = clamp_u8(org);
                px[2] = clamp_u8(orb);
                px[3] = clamp_u8(oaa);
            }
            pm = (npy_bool*)((char*)pm + smx);
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(img); Py_DECREF(msk);
    Py_RETURN_NONE;
}

/* ---------- sample_and_blend_color(image, x, y, color) -> (r,g,b,a) ---------- */
static PyObject *py_sample_and_blend_color(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *color_obj=NULL;
    int x, y;
    if (!PyArg_ParseTuple(args, "OiiO", &img_obj, &x, &y, &color_obj)) return NULL;

    unsigned char fr, fg, fb, fa;
    if (parse_rgba(color_obj, &fr, &fg, &fb, &fa) != 0) {
        PyErr_SetString(PyExc_TypeError, "foreground_color must be RGBA 4-tuple");
        return NULL;
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img) return NULL;

    if (PyArray_NDIM(img)!=3 || PyArray_DIM(img,2)!=4 || PyArray_TYPE(img)!=NPY_UBYTE) {
        Py_DECREF(img);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }

    const int H = (int)PyArray_DIM(img,0);
    const int W = (int)PyArray_DIM(img,1);

    if (x < 0 || x >= W || y < 0 || y >= H) {
        Py_DECREF(img);
        return Py_BuildValue("(iiii)", (int)fr, (int)fg, (int)fb, (int)fa);
    }

    unsigned char *px = (unsigned char*)PyArray_DATA(img) +
                        (npy_intp)y * PyArray_STRIDE(img,0) +
                        (npy_intp)x * PyArray_STRIDE(img,1);

    if (fa == 255) { Py_DECREF(img); return Py_BuildValue("(iiii)", (int)fr,(int)fg,(int)fb,255); }
    if (fa == 0)   { int br=px[0], bg=px[1], bb=px[2], ba=px[3]; Py_DECREF(img); return Py_BuildValue("(iiii)", br,bg,bb,ba); }

    const float a  = (float)fa / 255.0f;
    const float ia = 1.0f - a;

    const float br = (float)px[0];
    const float bg = (float)px[1];
    const float bb = (float)px[2];
    const float ba = (float)px[3];

    const int orr = (int)lroundf(fr * a + br * ia);
    const int org = (int)lroundf(fg * a + bg * ia);
    const int orb = (int)lroundf(fb * a + bb * ia);
    const int oaa = (int)lroundf((float)fa + ba * ia);

    Py_DECREF(img);
    return Py_BuildValue("(iiii)", clamp_u8(orr), clamp_u8(org), clamp_u8(orb), clamp_u8(oaa));
}

/* ---------- get_blended_color(x0,y0,x1,y1, image, color) -> (r,g,b,a) ---------- */
/* Samples 5px offsets at both endpoints; blends fg OVER those bg samples; averages. */
static PyObject *py_get_blended_color(PyObject *self, PyObject *args) {
    int x0, y0, x1, y1;
    PyObject *img_obj=NULL, *color_obj=NULL;
    if (!PyArg_ParseTuple(args, "iiiiOO", &x0, &y0, &x1, &y1, &img_obj, &color_obj)) return NULL;

    unsigned char fr, fg, fb, fa;
    if (parse_rgba(color_obj, &fr, &fg, &fb, &fa) != 0) {
        PyErr_SetString(PyExc_TypeError, "color must be RGBA 4-tuple");
        return NULL;
    }

    if (!img_obj || img_obj == Py_None) {
        return Py_BuildValue("(iiii)", (int)fr, (int)fg, (int)fb, (int)fa);
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img) return NULL;

    if (PyArray_NDIM(img)!=3 || PyArray_DIM(img,2)!=4 || PyArray_TYPE(img)!=NPY_UBYTE) {
        Py_DECREF(img);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }

    const int H = (int)PyArray_DIM(img,0);
    const int W = (int)PyArray_DIM(img,1);
    const npy_intp sy = PyArray_STRIDE(img,0);
    const npy_intp sx = PyArray_STRIDE(img,1);
    unsigned char *base = (unsigned char*)PyArray_DATA(img);

    if (fa == 255) { Py_DECREF(img); return Py_BuildValue("(iiii)", (int)fr,(int)fg,(int)fb,255); }
    if (fa == 0) {
        int mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
        if (mx < 0 || mx >= W || my < 0 || my >= H) { Py_DECREF(img); return Py_BuildValue("(iiii)", 0,0,0,0); }
        unsigned char *p = base + (npy_intp)my*sy + (npy_intp)mx*sx;
        Py_DECREF(img);
        return Py_BuildValue("(iiii)", (int)p[0], (int)p[1], (int)p[2], (int)p[3]);
    }

    const int dx = x1 - x0, dy = y1 - y0;
    const double len = sqrt((double)dx*dx + (double)dy*dy);
    const double off = 5.0;
    int sx0 = x0, sy0 = y0, sx1 = x1, sy1 = y1;
    if (len > 1e-6) {
        sx0 = (int)lrint((double)x0 - off * (double)dx / len);
        sy0 = (int)lrint((double)y0 - off * (double)dy / len);
        sx1 = (int)lrint((double)x1 + off * (double)dx / len);
        sy1 = (int)lrint((double)y1 + off * (double)dy / len);
    }

    int r0,g0,b0,a0, r1,g1,b1,a1;
    inline_blend_at(base, sy, sx, W, H, sx0, sy0, fr, fg, fb, fa, &r0,&g0,&b0,&a0);
    inline_blend_at(base, sy, sx, W, H, sx1, sy1, fr, fg, fb, fa, &r1,&g1,&b1,&a1);

    const int rr = clamp_u8((r0 + r1) / 2);
    const int rg = clamp_u8((g0 + g1) / 2);
    const int rb = clamp_u8((b0 + b1) / 2);
    const int ra = clamp_u8((a0 + a1) / 2);

    Py_DECREF(img);
    return Py_BuildValue("(iiii)", rr, rg, rb, ra);
}

/* ---------- methods / init ---------- */
static PyMethodDef Methods[] = {
    {"blend_mask_inplace", py_blend_mask_inplace, METH_VARARGS, "Blend RGBA into image wherever mask==True (straight alpha)."},
    {"sample_and_blend_color", py_sample_and_blend_color, METH_VARARGS, "Return fg OVER image[x,y] using straight alpha."},
    {"get_blended_color", py_get_blended_color, METH_VARARGS, "Segment-aware blended color (5px offset endpoints averaged)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_native", "mvcrender.blend", -1, Methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
