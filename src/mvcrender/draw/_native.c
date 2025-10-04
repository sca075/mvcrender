#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

/* ========================= Utilities ========================= */

static inline unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static int parse_rgba_tuple(PyObject *obj, unsigned char rgba[4]) {
    /* Accept (r,g,b) or (r,g,b,a) */
    if (!PyTuple_Check(obj)) return -1;
    Py_ssize_t n = PyTuple_Size(obj);
    if (n != 3 && n != 4) return -1;

    long r = PyLong_AsLong(PyTuple_GET_ITEM(obj, 0));
    long g = PyLong_AsLong(PyTuple_GET_ITEM(obj, 1));
    long b = PyLong_AsLong(PyTuple_GET_ITEM(obj, 2));
    if (PyErr_Occurred()) return -1;

    rgba[0] = clamp_u8((int)r);
    rgba[1] = clamp_u8((int)g);
    rgba[2] = clamp_u8((int)b);
    rgba[3] = 255;

    if (n == 4) {
        long a = PyLong_AsLong(PyTuple_GET_ITEM(obj, 3));
        if (PyErr_Occurred()) return -1;
        rgba[3] = clamp_u8((int)a);
    }
    return 0;
}

static inline void put_px_rgba(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int x, int y, const unsigned char rgba[4]
){
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
    unsigned char *p = base + (npy_intp)y * sy + (npy_intp)x * sx;
    p[0]=rgba[0]; p[1]=rgba[1]; p[2]=rgba[2]; p[3]=rgba[3];
}

/* Round disk brush for thickness (radius 0..25) */
static void build_disk_mask(int r, unsigned char *mask, int *side_out) {
    int side = 2*r + 1;
    *side_out = side;
    int rr = r*r;
    for (int yy=-r; yy<=r; ++yy) {
        for (int xx=-r; xx<=r; ++xx) {
            int idx = (yy + r)*side + (xx + r);
            mask[idx] = (xx*xx + yy*yy <= rr) ? 1 : 0;
        }
    }
}

static inline void stamp_disk_solid(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int cx, int cy, int r, const unsigned char rgba[4],
    const unsigned char *mask, int side
){
    int minx = cx - r, maxx = cx + r;
    int miny = cy - r, maxy = cy + r;

    int y0 = (miny < 0) ? 0 : miny;
    int y1 = (maxy >= H) ? (H-1) : maxy;
    int x0 = (minx < 0) ? 0 : minx;
    int x1 = (maxx >= W) ? (W-1) : maxx;

    for (int y=y0; y<=y1; ++y) {
        int my = y - (cy - r);
        unsigned char *row = base + (npy_intp)y * sy;
        for (int x=x0; x<=x1; ++x) {
            int mx = x - (cx - r);
            if (mask[my*side + mx]) {
                unsigned char *p = row + (npy_intp)x * sx;
                p[0]=rgba[0]; p[1]=rgba[1]; p[2]=rgba[2]; p[3]=rgba[3];
            }
        }
    }
}

/* ========================= Line (Bresenham) ========================= */

static void line_solid(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int x0, int y0, int x1, int y1,
    const unsigned char rgba[4], int thickness
){
    if (thickness < 1) thickness = 1;
    if (thickness > 25) thickness = 25;

    int r = thickness / 2;
    unsigned char mask[(2*25+1)*(2*25+1)];
    int side=0;
    build_disk_mask(r, mask, &side);

    int dx = abs(x1 - x0), sxn = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), syn = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        stamp_disk_solid(base, sy, sx, W, H, x0, y0, r, rgba, mask, side);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2*err;
        if (e2 >= dy) { err += dy; x0 += sxn; }
        if (e2 <= dx) { err += dx; y0 += syn; }
    }
}

/* ========================= Circle ========================= */

/* Midpoint circle points + stamping thickness by disk brush */
static void circle_outline_solid(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int cx, int cy, int radius,
    const unsigned char rgba[4], int thickness
){
    if (thickness < 1) thickness = 1;
    if (thickness > 25) thickness = 25;
    int rbrush = thickness / 2;
    unsigned char mask[(2*25+1)*(2*25+1)];
    int side=0;
    build_disk_mask(rbrush, mask, &side);

    int x = radius;
    int y = 0;
    int err = 1 - x;

    while (x >= y) {
        /* 8 octant points */
        stamp_disk_solid(base, sy, sx, W, H, cx + x, cy + y, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx + y, cy + x, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx - y, cy + x, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx - x, cy + y, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx - x, cy - y, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx - y, cy - x, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx + y, cy - x, rbrush, rgba, mask, side);
        stamp_disk_solid(base, sy, sx, W, H, cx + x, cy - y, rbrush, rgba, mask, side);

        y++;
        if (err < 0) err += 2*y + 1;
        else { x--; err += 2*(y - x + 1); }
    }
}

static void circle_filled_solid(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, int cx, int cy, int radius,
    const unsigned char rgba[4]
){
    if (radius < 0) return;
    for (int y = -radius; y <= radius; ++y) {
        int yy = cy + y;
        if ((unsigned)yy >= (unsigned)H) continue;
        /* horizontal span length */
        int dx = (int)floor(sqrt((double)radius*radius - (double)y*y));
        int xstart = cx - dx;
        int xend   = cx + dx;
        if (xstart < 0) xstart = 0;
        if (xend >= W)  xend = W-1;

        unsigned char *row = base + (npy_intp)yy * sy + (npy_intp)xstart * sx;
        for (int x=xstart; x<=xend; ++x) {
            row[0]=rgba[0]; row[1]=rgba[1]; row[2]=rgba[2]; row[3]=rgba[3];
            row += sx;
        }
    }
}

/* ========================= Polygon ========================= */
/* Scanline fill for simple polygons, y in [minY,maxY] inclusive */
static void polygon_fill_solid(
    unsigned char *base, npy_intp sy, npy_intp sx,
    int W, int H, const int *xs, const int *ys, int n,
    const unsigned char rgba[4]
){
    if (n < 3) return;

    /* Find vertical bounds */
    int minY = ys[0], maxY = ys[0];
    for (int i=1; i<n; ++i) {
        if (ys[i] < minY) minY = ys[i];
        if (ys[i] > maxY) maxY = ys[i];
    }
    if (minY < 0) minY = 0;
    if (maxY >= H) maxY = H-1;
    if (minY > maxY) return;

    for (int y = minY; y <= maxY; ++y) {
        /* Build intersection list */
        int inter_count = 0;
        int inter_x[4096]; /* plenty for our typical small polygons */
        for (int i=0, j=n-1; i<n; j=i++) {
            int yi = ys[i], yj = ys[j];
            int xi = xs[i], xj = xs[j];

            /* check if edge crosses scanline y */
            bool cond1 = (yi <= y && y < yj);
            bool cond2 = (yj <= y && y < yi);
            if (cond1 || cond2) {
                double t = (double)(y - yi) / (double)(yj - yi);
                int xint = (int)floor(xi + t * (xj - xi) + 0.5);
                if (inter_count < (int)(sizeof(inter_x)/sizeof(inter_x[0])))
                    inter_x[inter_count++] = xint;
            }
        }
        if (inter_count < 2) continue;

        /* sort intersections (small n; insertion sort) */
        for (int i=1; i<inter_count; ++i) {
            int key = inter_x[i], k=i-1;
            while (k>=0 && inter_x[k] > key) { inter_x[k+1]=inter_x[k]; --k; }
            inter_x[k+1] = key;
        }

        /* fill pairs */
        for (int k=0; k+1<inter_count; k+=2) {
            int x0 = inter_x[k];
            int x1 = inter_x[k+1];
            if (x0 > x1) { int t=x0; x0=x1; x1=t; }
            if (x1 < 0 || x0 >= W) continue;
            if (x0 < 0) x0 = 0;
            if (x1 >= W) x1 = W-1;

            unsigned char *row = base + (npy_intp)y * sy + (npy_intp)x0 * sx;
            for (int x=x0; x<=x1; ++x) {
                row[0]=rgba[0]; row[1]=rgba[1]; row[2]=rgba[2]; row[3]=rgba[3];
                row += sx;
            }
        }
    }
}

/* ========================= Python bindings ========================= */

static int ensure_img_rgba(PyArrayObject *img) {
    return !(img &&
             PyArray_NDIM(img)==3 &&
             PyArray_TYPE(img)==NPY_UBYTE &&
             PyArray_DIM(img,2)==4 &&
             PyArray_ISCARRAY(img));
}

/* line_u8(image, x0,y0,x1,y1, color, thickness) -> (x0,y0,x1,y1) */
static PyObject *py_line_u8(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *color_obj=NULL;
    int x0, y0, x1, y1, thickness;

    if (!PyArg_ParseTuple(args, "OiiiiOi", &img_obj, &x0, &y0, &x1, &y1, &color_obj, &thickness))
        return NULL;

    unsigned char rgba[4];
    if (parse_rgba_tuple(color_obj, rgba) != 0) {
        PyErr_SetString(PyExc_TypeError, "color must be (r,g,b) or (r,g,b,a)");
        return NULL;
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img) return NULL;
    if (ensure_img_rgba(img)) {
        Py_DECREF(img);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }

    int H = (int)PyArray_DIM(img,0);
    int W = (int)PyArray_DIM(img,1);
    npy_intp sy = PyArray_STRIDE(img,0);
    npy_intp sx = PyArray_STRIDE(img,1);
    unsigned char *base = (unsigned char*)PyArray_DATA(img);

    if (thickness < 1) thickness = 1;
    if (thickness > 25) thickness = 25;

    line_solid(base, sy, sx, W, H, x0, y0, x1, y1, rgba, thickness);

    /* ROI (approx): bbox of the segment expanded by thickness */
    int r = thickness/2;
    int rx0 = x0 < x1 ? x0 : x1;
    int ry0 = y0 < y1 ? y0 : y1;
    int rx1 = x0 > x1 ? x0 : x1;
    int ry1 = y0 > y1 ? y0 : y1;
    rx0 -= r; ry0 -= r; rx1 += r; ry1 += r;
    if (rx0 < 0) rx0 = 0; if (ry0 < 0) ry0 = 0;
    if (rx1 >= W) rx1 = W-1; if (ry1 >= H) ry1 = H-1;

    Py_DECREF(img);
    return Py_BuildValue("(iiii)", rx0, ry0, rx1, ry1);
}

/* polyline_u8(image, xs, ys, thickness, color) -> (x0,y0,x1,y1) */
static PyObject *py_polyline_u8(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *xs_obj=NULL, *ys_obj=NULL, *color_obj=NULL;
    int thickness;

    if (!PyArg_ParseTuple(args, "OOOOi", &img_obj, &xs_obj, &ys_obj, &color_obj, &thickness))
        return NULL;

    unsigned char rgba[4];
    if (parse_rgba_tuple(color_obj, rgba) != 0) {
        PyErr_SetString(PyExc_TypeError, "color must be (r,g,b) or (r,g,b,a)");
        return NULL;
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *xs  = (PyArrayObject*)PyArray_FROM_OTF(xs_obj, NPY_INT32,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *ys  = (PyArrayObject*)PyArray_FROM_OTF(ys_obj, NPY_INT32,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img || !xs || !ys) { Py_XDECREF(img); Py_XDECREF(xs); Py_XDECREF(ys); return NULL; }
    if (ensure_img_rgba(img)) {
        Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }
    if (PyArray_NDIM(xs)!=1 || PyArray_NDIM(ys)!=1 || PyArray_DIM(xs,0)!=PyArray_DIM(ys,0)) {
        Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
        PyErr_SetString(PyExc_ValueError, "xs and ys must be 1D int32 arrays of same length");
        return NULL;
    }

    int H = (int)PyArray_DIM(img,0);
    int W = (int)PyArray_DIM(img,1);
    npy_intp sy = PyArray_STRIDE(img,0);
    npy_intp sx = PyArray_STRIDE(img,1);
    unsigned char *base = (unsigned char*)PyArray_DATA(img);

    if (thickness < 1) thickness = 1;
    if (thickness > 25) thickness = 25;

    int *xv = (int*)PyArray_DATA(xs);
    int *yv = (int*)PyArray_DATA(ys);
    npy_intp N = PyArray_DIM(xs,0);
    if (N < 2) {
        Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
        return Py_BuildValue("(iiii)", 0,0,-1,-1);
    }

    int rx0= W-1, ry0= H-1, rx1=0, ry1=0;
    for (npy_intp i=0; i<N-1; ++i) {
        int x0=xv[i], y0=yv[i], x1=xv[i+1], y1=yv[i+1];
        line_solid(base, sy, sx, W, H, x0, y0, x1, y1, rgba, thickness);
        if (x0 < rx0) rx0 = x0; if (x1 < rx0) rx0 = x1;
        if (y0 < ry0) ry0 = y0; if (y1 < ry0) ry0 = y1;
        if (x0 > rx1) rx1 = x0; if (x1 > rx1) rx1 = x1;
        if (y0 > ry1) ry1 = y0; if (y1 > ry1) ry1 = y1;
    }
    int r = thickness/2;
    rx0 -= r; ry0 -= r; rx1 += r; ry1 += r;
    if (rx0 < 0) rx0 = 0; if (ry0 < 0) ry0 = 0;
    if (rx1 >= W) rx1 = W-1; if (ry1 >= H) ry1 = H-1;

    Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
    return Py_BuildValue("(iiii)", rx0, ry0, rx1, ry1);
}

/* circle_u8(image, cx,cy, radius, color, thickness) -> (x0,y0,x1,y1)
   thickness < 0 => filled; thickness >=1 => outlined stroke of that width */
static PyObject *py_circle_u8(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *color_obj=NULL;
    int cx, cy, radius, thickness;

    if (!PyArg_ParseTuple(args, "OiiiOi", &img_obj, &cx, &cy, &radius, &color_obj, &thickness))
        return NULL;

    if (radius < 0) radius = 0;
    if (radius > 4096) radius = 4096; /* sanity */

    unsigned char rgba[4];
    if (parse_rgba_tuple(color_obj, rgba) != 0) {
        PyErr_SetString(PyExc_TypeError, "color must be (r,g,b) or (r,g,b,a)");
        return NULL;
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img) return NULL;
    if (ensure_img_rgba(img)) {
        Py_DECREF(img);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }

    int H = (int)PyArray_DIM(img,0);
    int W = (int)PyArray_DIM(img,1);
    npy_intp sy = PyArray_STRIDE(img,0);
    npy_intp sx = PyArray_STRIDE(img,1);
    unsigned char *base = (unsigned char*)PyArray_DATA(img);

    if (thickness < 0) {
        circle_filled_solid(base, sy, sx, W, H, cx, cy, radius, rgba);
    } else {
        if (thickness < 1) thickness = 1;
        if (thickness > 25) thickness = 25;
        circle_outline_solid(base, sy, sx, W, H, cx, cy, radius, rgba, thickness);
    }

    int pad = (thickness < 0) ? 0 : (thickness/2);
    int x0 = cx - radius - pad, x1 = cx + radius + pad;
    int y0 = cy - radius - pad, y1 = cy + radius + pad;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;

    Py_DECREF(img);
    return Py_BuildValue("(iiii)", x0,y0,x1,y1);
}

/* polygon_u8(image, xs, ys, thickness, outline_color, fill_color_or_None) -> (x0,y0,x1,y1) */
static PyObject *py_polygon_u8(PyObject *self, PyObject *args) {
    PyObject *img_obj=NULL, *xs_obj=NULL, *ys_obj=NULL, *out_color_obj=NULL, *fill_color_obj=NULL;
    int thickness;

    if (!PyArg_ParseTuple(args, "OOOOiO", &img_obj, &xs_obj, &ys_obj, &out_color_obj, &thickness, &fill_color_obj))
        return NULL;

    unsigned char out_rgba[4], fill_rgba[4];
    if (parse_rgba_tuple(out_color_obj, out_rgba) != 0) {
        PyErr_SetString(PyExc_TypeError, "outline_color must be (r,g,b) or (r,g,b,a)");
        return NULL;
    }
    bool do_fill = (fill_color_obj != Py_None);
    if (do_fill) {
        if (parse_rgba_tuple(fill_color_obj, fill_rgba) != 0) {
            PyErr_SetString(PyExc_TypeError, "fill_color must be (r,g,b) or (r,g,b,a) or None");
            return NULL;
        }
    }

    PyArrayObject *img = (PyArrayObject*)PyArray_FROM_OTF(img_obj, NPY_UINT8,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *xs  = (PyArrayObject*)PyArray_FROM_OTF(xs_obj, NPY_INT32,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *ys  = (PyArrayObject*)PyArray_FROM_OTF(ys_obj, NPY_INT32,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!img || !xs || !ys) { Py_XDECREF(img); Py_XDECREF(xs); Py_XDECREF(ys); return NULL; }
    if (ensure_img_rgba(img)) {
        Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
        PyErr_SetString(PyExc_ValueError, "image must be HxWx4 uint8 C-contiguous");
        return NULL;
    }
    if (PyArray_NDIM(xs)!=1 || PyArray_NDIM(ys)!=1 || PyArray_DIM(xs,0)!=PyArray_DIM(ys,0) || PyArray_DIM(xs,0)<3) {
        Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
        PyErr_SetString(PyExc_ValueError, "xs and ys must be 1D int32 arrays with length >=3 and same length");
        return NULL;
    }

    int H = (int)PyArray_DIM(img,0);
    int W = (int)PyArray_DIM(img,1);
    npy_intp sy = PyArray_STRIDE(img,0);
    npy_intp sx = PyArray_STRIDE(img,1);
    unsigned char *base = (unsigned char*)PyArray_DATA(img);

    int *xv = (int*)PyArray_DATA(xs);
    int *yv = (int*)PyArray_DATA(ys);
    int n = (int)PyArray_DIM(xs,0);

    /* Fill first (if requested), then outline on top */
    if (do_fill) {
        polygon_fill_solid(base, sy, sx, W, H, xv, yv, n, fill_rgba);
    }

    if (thickness < 1) thickness = 1;
    if (thickness > 25) thickness = 25;

    for (int i=0; i<n; ++i) {
        int j = (i+1) % n;
        line_solid(base, sy, sx, W, H, xv[i], yv[i], xv[j], yv[j], out_rgba, thickness);
    }

    /* ROI: bbox of polygon plus thickness pad */
    int minx=xv[0], maxx=xv[0], miny=yv[0], maxy=yv[0];
    for (int i=1;i<n;++i) {
        if (xv[i]<minx) minx=xv[i];
        if (xv[i]>maxx) maxx=xv[i];
        if (yv[i]<miny) miny=yv[i];
        if (yv[i]>maxy) maxy=yv[i];
    }
    int pad = thickness/2;
    minx -= pad; miny -= pad; maxx += pad; maxy += pad;
    if (minx < 0) minx = 0; if (miny < 0) miny = 0;
    if (maxx >= W) maxx = W-1; if (maxy >= H) maxy = H-1;

    Py_DECREF(img); Py_DECREF(xs); Py_DECREF(ys);
    return Py_BuildValue("(iiii)", minx, miny, maxx, maxy);
}

/* -------------------- Module -------------------- */

static PyMethodDef Methods[] = {
    {"line_u8",     py_line_u8,     METH_VARARGS, "Draw a solid line with thickness (overwrite RGBA)."},
    {"polyline_u8", py_polyline_u8, METH_VARARGS, "Draw a solid polyline with thickness (overwrite RGBA)."},
    {"circle_u8",   py_circle_u8,   METH_VARARGS, "Draw a circle. thickness<0 => filled; else outlined."},
    {"polygon_u8",  py_polygon_u8,  METH_VARARGS, "Draw polygon outline (thickness) and optional fill."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_native", "mvcrender.draw (no-blend raster)", -1, Methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
