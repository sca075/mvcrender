// src/mvcrender/autocrop/_native.c
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

/*
  1:1 method names with your Python AutoCrop:
    - validate_crop_dimensions(shared) -> bool
    - check_trim(trimmed_height, trimmed_width, margin_size, image_array, file_name, rotate)
        * raises TrimError(message, image=image_array) on failure
    - _calculate_trimmed_dimensions() -> (trimmed_width, trimmed_height) and updates shared.image_ref_*
    - _async_auto_crop_data(tdata) -> list|None
    - auto_crop_offset() -> None
    - _init_auto_crop() -> list|None
    - async_image_margins(image_array, detect_colour) -> (min_y, min_x, max_x, max_y)
    - async_get_room_bounding_box(room_name, rand256=False) -> tuple|None
    - async_check_if_zoom_is_on(image_array, margin_size=100, zoom=False, rand256=False) -> ndarray
    - async_rotate_the_image(trimmed, rotate) -> ndarray
    - auto_trim_and_zoom_image(image_array, detect_colour=(93,109,126,255),
                                     margin_size=0, rotate=0, zoom=False, rand256=False) -> ndarray
*/

typedef struct {
    PyObject_HEAD
    PyObject* handler;
    PyObject* auto_crop;        // list [l,u,r,d] or None
    PyObject* crop_area;        // list or None

    long trim_up, trim_down, trim_left, trim_right;
    long offset_top, offset_bottom, offset_left, offset_right;
} AutoCropObject;

static PyObject* TrimError = NULL;

/* ------------------------- small helpers ------------------------- */

static inline long clampl(long v, long lo, long hi){
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static PyObject* make_list4(long a,long b,long c,long d){
    PyObject* l=PyList_New(4);
    if(!l) return NULL;
    PyList_SET_ITEM(l,0,PyLong_FromLong(a));
    PyList_SET_ITEM(l,1,PyLong_FromLong(b));
    PyList_SET_ITEM(l,2,PyLong_FromLong(c));
    PyList_SET_ITEM(l,3,PyLong_FromLong(d));
    return l;
}

static int as_rgba_nd(uint8_t** data, int* H, int* W, PyArrayObject** arr_out, PyObject* obj) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_ARRAY_CARRAY);
    if (!arr) return -1;
    if (PyArray_NDIM(arr) != 3 || PyArray_DIM(arr,2) != 4) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Expected RGBA ndarray HxWx4 uint8");
        return -1;
    }
    *H = (int)PyArray_DIM(arr,0);
    *W = (int)PyArray_DIM(arr,1);
    *data = (uint8_t*)PyArray_DATA(arr);
    *arr_out = arr;
    return 0;
}

static int load_trims_offsets(AutoCropObject* self) {
    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    if (!shared) return -1;

    // trims
    PyObject* trims = PyObject_GetAttrString(shared, "trims");
    if (!trims) { Py_DECREF(shared); return -1; }
    PyObject* to_dict = PyObject_GetAttrString(trims, "to_dict");
    if (!to_dict) { Py_DECREF(trims); Py_DECREF(shared); return -1; }
    PyObject* dict = PyObject_CallObject(to_dict, NULL);
    Py_DECREF(to_dict);
    Py_DECREF(trims);
    if (!dict) { Py_DECREF(shared); return -1; }

    long tu=0, td=0, tl=0, tr=0;
    PyObject* v;
    v = PyDict_GetItemString(dict, "trim_up");
    if (v && v != Py_None) { tu = PyLong_AsLong(v); if (PyErr_Occurred()) { Py_DECREF(dict); Py_DECREF(shared); return -1; } }
    v = PyDict_GetItemString(dict, "trim_down");
    if (v && v != Py_None) { td = PyLong_AsLong(v); if (PyErr_Occurred()) { Py_DECREF(dict); Py_DECREF(shared); return -1; } }
    v = PyDict_GetItemString(dict, "trim_left");
    if (v && v != Py_None) { tl = PyLong_AsLong(v); if (PyErr_Occurred()) { Py_DECREF(dict); Py_DECREF(shared); return -1; } }
    v = PyDict_GetItemString(dict, "trim_right");
    if (v && v != Py_None) { tr = PyLong_AsLong(v); if (PyErr_Occurred()) { Py_DECREF(dict); Py_DECREF(shared); return -1; } }
    Py_DECREF(dict);

    self->trim_up = tu; self->trim_down = td; self->trim_left = tl; self->trim_right = tr;

    // offsets - with NULL checks
    PyObject* off_top    = PyObject_GetAttrString(shared, "offset_top");
    PyObject* off_down   = PyObject_GetAttrString(shared, "offset_down");
    PyObject* off_left   = PyObject_GetAttrString(shared, "offset_left");
    PyObject* off_right  = PyObject_GetAttrString(shared, "offset_right");

    self->offset_top = 0;
    if (off_top && off_top != Py_None) {
        self->offset_top = PyLong_AsLong(off_top);
        if (PyErr_Occurred()) { PyErr_Clear(); self->offset_top = 0; }
    }

    self->offset_bottom = 0;
    if (off_down && off_down != Py_None) {
        self->offset_bottom = PyLong_AsLong(off_down);
        if (PyErr_Occurred()) { PyErr_Clear(); self->offset_bottom = 0; }
    }

    self->offset_left = 0;
    if (off_left && off_left != Py_None) {
        self->offset_left = PyLong_AsLong(off_left);
        if (PyErr_Occurred()) { PyErr_Clear(); self->offset_left = 0; }
    }

    self->offset_right = 0;
    if (off_right && off_right != Py_None) {
        self->offset_right = PyLong_AsLong(off_right);
        if (PyErr_Occurred()) { PyErr_Clear(); self->offset_right = 0; }
    }

    Py_XDECREF(off_top); Py_XDECREF(off_down); Py_XDECREF(off_left); Py_XDECREF(off_right);

    Py_DECREF(shared);
    return 0;
}

static void update_shared_ref_dims(AutoCropObject* self, long tw, long th){
    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    if (!shared) return;
    PyObject* wv = PyLong_FromLong(tw);
    PyObject* hv = PyLong_FromLong(th);
    if (wv && hv) {
        PyObject_SetAttrString(shared, "image_ref_width", wv);
        PyObject_SetAttrString(shared, "image_ref_height", hv);
    }
    Py_XDECREF(wv); Py_XDECREF(hv);
    Py_DECREF(shared);
}

/* ---------------------- bbox / crop / rotate ---------------------- */

static int bbox_from_alpha(uint8_t* rgba, int H, int W, int threshold, int* x0,int* y0,int* x1,int* y1){
    int xmin=W, ymin=H, xmax=-1, ymax=-1;
    for (int y=0; y<H; y++){
        uint8_t* row = rgba + ((size_t)y*(size_t)W)*4;
        for (int x=0; x<W; x++){
            uint8_t a = row[(size_t)x*4 + 3];
            if (a >= threshold){
                if (x < xmin) xmin = x;
                if (y < ymin) ymin = y;
                if (x > xmax) xmax = x;
                if (y > ymax) ymax = y;
            }
        }
    }
    if (xmax < 0) return 0;
    *x0=xmin; *y0=ymin; *x1=xmax; *y1=ymax;
    return 1;
}

static int bbox_from_color(uint8_t* rgba, int H, int W, const uint8_t dc[4], int* x0,int* y0,int* x1,int* y1){
    int xmin=W, ymin=H, xmax=-1, ymax=-1;
    for (int y=0; y<H; y++){
        uint8_t* row = rgba + ((size_t)y*(size_t)W)*4;
        for (int x=0; x<W; x++){
            size_t i = (size_t)x*4;
            uint8_t r=row[i], g=row[i+1], b=row[i+2], a=row[i+3];
            if (!(r==dc[0] && g==dc[1] && b==dc[2] && a==dc[3])){
                if (x < xmin) xmin = x;
                if (y < ymin) ymin = y;
                if (x > xmax) xmax = x;
                if (y > ymax) ymax = y;
            }
        }
    }
    if (xmax < 0) return 0;
    *x0=xmin; *y0=ymin; *x1=xmax; *y1=ymax;
    return 1;
}

static PyObject* crop_rgba(PyArrayObject* src_arr, int x0,int y0,int x1,int y1){
    int H = (int)PyArray_DIM(src_arr,0);
    int W = (int)PyArray_DIM(src_arr,1);
    (void)H;

    int cw = x1 - x0;
    int ch = y1 - y0;
    if (cw <= 0 || ch <= 0){
        PyErr_SetString(PyExc_ValueError, "Invalid crop region");
        return NULL;
    }

    npy_intp dims[3] = { ch, cw, 4 };
    PyObject* out = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!out) return NULL;

    uint8_t* src = (uint8_t*)PyArray_DATA(src_arr);
    uint8_t* dst = (uint8_t*)PyArray_DATA((PyArrayObject*)out);

    size_t src_stride = (size_t)W*4;
    size_t dst_stride = (size_t)cw*4;

    Py_BEGIN_ALLOW_THREADS
    for (int yy=0; yy<ch; yy++){
        uint8_t* srow = src + ((size_t)(y0+yy) * src_stride) + (size_t)x0 * 4;
        uint8_t* drow = dst + ((size_t)yy * dst_stride);
        memcpy(drow, srow, dst_stride);
    }
    Py_END_ALLOW_THREADS
    return out;
}

static PyObject* rot_rgba(PyArrayObject* src_arr, int rotate){
    int H = (int)PyArray_DIM(src_arr,0);
    int W = (int)PyArray_DIM(src_arr,1);
    uint8_t* src = (uint8_t*)PyArray_DATA(src_arr);

    npy_intp dims90[3] = { W, H, 4 };
    npy_intp dims00[3] = { H, W, 4 };
    PyObject* out = NULL;

    if (rotate == 0){
        out = PyArray_SimpleNew(3, dims00, NPY_UINT8);
        if (!out) return NULL;
        Py_BEGIN_ALLOW_THREADS
        memcpy(PyArray_DATA((PyArrayObject*)out), src, (size_t)H*(size_t)W*4);
        Py_END_ALLOW_THREADS
        return out;
    } else if (rotate == 90 || rotate == 270){
        out = PyArray_SimpleNew(3, dims90, NPY_UINT8);
    } else if (rotate == 180){
        out = PyArray_SimpleNew(3, dims00, NPY_UINT8);
    } else {
        PyErr_SetString(PyExc_ValueError, "rotate must be 0, 90, 180, or 270");
        return NULL;
    }
    if (!out) return NULL;

    uint8_t* dst = (uint8_t*)PyArray_DATA((PyArrayObject*)out);

    if (rotate == 180){
        Py_BEGIN_ALLOW_THREADS
        for (int y=0; y<H; y++){
            for (int x=0; x<W; x++){
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                size_t di = ((size_t)(H-1-y) * (size_t)W + (size_t)(W-1-x)) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
        Py_END_ALLOW_THREADS
    } else if (rotate == 270){
        int OW = H;
        Py_BEGIN_ALLOW_THREADS
        for (int y=0; y<H; y++){
            for (int x=0; x<W; x++){
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                int ny = x;
                int nx = OW - 1 - y;
                size_t di = ((size_t)ny * (size_t)OW + (size_t)nx) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
        Py_END_ALLOW_THREADS
    } else { // 90
        int OW = H;
        Py_BEGIN_ALLOW_THREADS
        for (int y=0; y<H; y++){
            for (int x=0; x<W; x++){
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                int ny = W - 1 - x;
                int nx = y;
                size_t di = ((size_t)ny * (size_t)OW + (size_t)nx) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
        Py_END_ALLOW_THREADS
    }
    return out;
}

/* --------------------------- object basics --------------------------- */

static int AutoCrop_init(AutoCropObject* self, PyObject* args, PyObject* kw){
    PyObject* handler = NULL;
    static char* kwlist[] = {"handler", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", kwlist, &handler)) return -1;
    Py_INCREF(handler);
    self->handler = handler;
    self->auto_crop = Py_None; Py_INCREF(Py_None);
    self->crop_area = Py_None; Py_INCREF(Py_None);
    if (load_trims_offsets(self) != 0) return -1;
    return 0;
}

static void AutoCrop_dealloc(AutoCropObject* self){
    Py_XDECREF(self->handler);
    Py_XDECREF(self->auto_crop);
    Py_XDECREF(self->crop_area);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* --------------------------- methods 1:1 --------------------------- */

// validate_crop_dimensions(shared) -> bool
static PyObject* AutoCrop_validate_crop_dimensions(AutoCropObject* self, PyObject* args){
    PyObject* shared;
    if (!PyArg_ParseTuple(args, "O", &shared)) return NULL;
    PyObject* w = PyObject_GetAttrString(shared, "image_ref_width");
    PyObject* h = PyObject_GetAttrString(shared, "image_ref_height");
    int ok = 1;
    if (!w || !h) ok = 0;
    else {
        long W = PyLong_AsLong(w);
        long H = PyLong_AsLong(h);
        if (W <= 0 || H <= 0) ok = 0;
    }
    Py_XDECREF(w); Py_XDECREF(h);
    if (!ok){
        PyObject* logger = PyImport_ImportModule("logging");
        if (logger){
            PyObject* getLogger = PyObject_GetAttrString(logger, "getLogger");
            if (getLogger){
                PyObject* l = PyObject_CallFunction(getLogger, "s", "mvcrender.autocrop");
                if (l){
                    PyObject* warn = PyObject_GetAttrString(l, "warning");
                    if (warn){
                        PyObject* _ = PyObject_CallFunction(warn,
                            "sll", "Auto-crop failed: Invalid dimensions (width=%s, height=%s). Using original image.", 0L, 0L);
                        Py_XDECREF(_);
                    }
                    Py_DECREF(l);
                }
                Py_DECREF(getLogger);
            }
            Py_DECREF(logger);
        }
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

// check_trim(trimmed_height, trimmed_width, margin_size, image_array, file_name, rotate)
static PyObject* AutoCrop_check_trim(AutoCropObject* self, PyObject* args){
    long th=0, tw=0, margin=0, rotate=0;
    PyObject* image_array=NULL; PyObject* file_name=NULL;
    if (!PyArg_ParseTuple(args, "lllOl", &th, &tw, &margin, &image_array, &rotate))
        return NULL;

    if (th <= margin || tw <= margin){
        // self.crop_area = [0,0,W,H]  and handler.img_size = (W,H)
        PyArrayObject* arr; uint8_t* p; int H,W;
        if (as_rgba_nd(&p, &H, &W, &arr, image_array) != 0) return NULL;

        Py_XDECREF(self->crop_area);
        self->crop_area = make_list4(0,0,W,H);
        // sync to handler for Python parity
        if (self->handler && self->crop_area){
            PyObject_SetAttrString(self->handler, "crop_area", self->crop_area);
        }

        PyObject* size_tuple = Py_BuildValue("(ii)", W, H);
        PyObject_SetAttrString(self->handler, "img_size", size_tuple);
        Py_DECREF(size_tuple);

        // raise TrimError with image attached
        PyObject* args_te = Py_BuildValue("(s)", "Trimming failed.");
        PyObject* exc = PyObject_CallObject(TrimError, args_te);
        Py_DECREF(args_te);
        if (exc){
            PyObject_SetAttrString(exc, "image", (PyObject*)arr); // arr is still a valid ndarray
            Py_DECREF(arr);
            PyErr_SetObject(TrimError, exc);
            Py_DECREF(exc);
            return NULL;
        } else {
            Py_DECREF(arr);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

// _calculate_trimmed_dimensions() -> (trimmed_width, trimmed_height)
static PyObject* AutoCrop__calculate_trimmed_dimensions(AutoCropObject* self, PyObject* Py_UNUSED(ignored)){
    long tw = (self->trim_right - self->offset_right) - (self->trim_left + self->offset_left);
    long th = (self->trim_down  - self->offset_bottom) - (self->trim_up   + self->offset_top);
    if (tw < 1) tw = 1;
    if (th < 1) th = 1;
    update_shared_ref_dims(self, tw, th);
    return Py_BuildValue("(ll)", tw, th);
}

// _async_auto_crop_data(tdata) -> list|None
static PyObject* AutoCrop__async_auto_crop_data(AutoCropObject* self, PyObject* args){
    PyObject* tdata;
    if (!PyArg_ParseTuple(args, "O", &tdata)) return NULL;

    if (self->auto_crop && self->auto_crop != Py_None){
        Py_RETURN_NONE;
    }

    // trims_data = TrimCropData.from_dict(dict(tdata.to_dict())).to_list()
    PyObject* to_dict = PyObject_GetAttrString(tdata, "to_dict");
    if (!to_dict) Py_RETURN_NONE;
    PyObject* tdict = PyObject_CallObject(to_dict, NULL);
    Py_DECREF(to_dict);
    if (!tdict) Py_RETURN_NONE;

    // expect python will give ints
    PyObject* tl = PyDict_GetItemString(tdict, "trim_left");
    PyObject* tu = PyDict_GetItemString(tdict, "trim_up");
    PyObject* tr = PyDict_GetItemString(tdict, "trim_right");
    PyObject* td = PyDict_GetItemString(tdict, "trim_down");

    long l = tl ? PyLong_AsLong(tl) : 0;
    long u = tu ? PyLong_AsLong(tu) : 0;
    long r = tr ? PyLong_AsLong(tr) : 0;
    long d = td ? PyLong_AsLong(td) : 0;

    Py_DECREF(tdict);

    self->trim_left=l; self->trim_up=u; self->trim_right=r; self->trim_down=d;

    if (!(l==0 && u==0 && r==0 && d==0)){
        (void)AutoCrop__calculate_trimmed_dimensions(self, NULL);
        Py_XDECREF(self->auto_crop);
        self->auto_crop = make_list4(l,u,r,d);
        Py_INCREF(self->auto_crop);
        return self->auto_crop;
    }
    Py_RETURN_NONE;
}

// auto_crop_offset()
static PyObject* AutoCrop_auto_crop_offset(AutoCropObject* self, PyObject* Py_UNUSED(ignored)){
    if (self->auto_crop && self->auto_crop != Py_None){
        PyObject* l = PyList_GetItem(self->auto_crop, 0);
        PyObject* u = PyList_GetItem(self->auto_crop, 1);
        PyObject* r = PyList_GetItem(self->auto_crop, 2);
        PyObject* d = PyList_GetItem(self->auto_crop, 3);
        long nl = PyLong_AsLong(l) + self->offset_left;
        long nu = PyLong_AsLong(u) + self->offset_top;
        long nr = PyLong_AsLong(r) - self->offset_right;
        long nd = PyLong_AsLong(d) - self->offset_bottom;
        Py_DECREF(self->auto_crop);
        self->auto_crop = make_list4(nl,nu,nr,nd);
    }
    Py_RETURN_NONE;
}

// _init_auto_crop() -> list|None
static PyObject* AutoCrop__init_auto_crop(AutoCropObject* self, PyObject* Py_UNUSED(ignored)){
    if (!self->auto_crop || self->auto_crop == Py_None){
        // compute from shared trims
        if (load_trims_offsets(self) != 0){
            Py_RETURN_NONE;
        }
        if (self->trim_left || self->trim_up || self->trim_right || self->trim_down){
            Py_XDECREF(self->auto_crop);
            self->auto_crop = make_list4(self->trim_left, self->trim_up, self->trim_right, self->trim_down);
            (void)AutoCrop_auto_crop_offset(self, NULL);
            // Update shared ref dimensions when using predefined trims
            (void)AutoCrop__calculate_trimmed_dimensions(self, NULL);
        } else {
            Py_RETURN_NONE;
        }
    } else {
        // preserve your behavior
        PyObject* v = PyLong_FromLong(1205);
        if (v){
            PyObject_SetAttrString(self->handler, "max_frames", v);
            Py_DECREF(v);
        }
    }
    // validate
    int bad = 0;
    if (!self->auto_crop || self->auto_crop == Py_None) bad = 1;
    else {
        for (int i=0;i<4;i++){
            PyObject* it = PyList_GetItem(self->auto_crop, i);
            if (!it || PyLong_AsLong(it) < 0){ bad = 1; break; }
        }
    }
    if (bad){
        Py_XDECREF(self->auto_crop);
        self->auto_crop = Py_None; Py_INCREF(Py_None);
        Py_RETURN_NONE;
    }
    Py_INCREF(self->auto_crop);
    return self->auto_crop;
}

/* async_image_margins(image_array, detect_colour) -> (min_y, min_x, max_x, max_y) */
static PyObject* AutoCrop_async_image_margins(AutoCropObject* self, PyObject* args){
    PyObject* img_obj; PyObject* col;
    if (!PyArg_ParseTuple(args, "OO", &img_obj, &col)) return NULL;

    PyArrayObject* img_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &img_arr, img_obj) != 0) return NULL;

    uint8_t dc[4] = {93,109,126,255};
    if (PySequence_Check(col) && PySequence_Size(col)==4){
        for (int i=0;i<4;i++){
            PyObject* it = PySequence_GetItem(col, i);
            dc[i] = (uint8_t)PyLong_AsLong(it);
            Py_XDECREF(it);
        }
    }

    int x0,y0,x1,y1, ok=0;
    Py_BEGIN_ALLOW_THREADS
    if (dc[3]==0) ok = bbox_from_alpha(p,H,W,1,&x0,&y0,&x1,&y1);
    else          ok = bbox_from_color(p,H,W,dc,&x0,&y0,&x1,&y1);
    Py_END_ALLOW_THREADS

    Py_DECREF(img_arr);

    if (!ok){
        // fallback: full
        return Py_BuildValue("(iiii)", 0, 0, W-1, H-1);
    }
    // (min_y, min_x, max_x, max_y)
    return Py_BuildValue("(iiii)", y0, x0, x1, y1);
}

/* async_get_room_bounding_box(room_name, rand256=False) -> tuple|None */
static PyObject* AutoCrop_async_get_room_bounding_box(AutoCropObject* self, PyObject* args, PyObject* kw){
    PyObject* room_name; int rand256 = 0;
    static char* kwlist[] = {"room_name", "rand256", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|p", kwlist, &room_name, &rand256)) return NULL;

    // room_propriety first
    PyObject* rp = PyObject_GetAttrString(self->handler, "room_propriety");
    if (rp && rp != Py_None){
        PyObject* room_dict = NULL;
        if (PyDict_Check(rp)) room_dict = rp;
        else if (PyTuple_Check(rp) && PyTuple_GET_SIZE(rp)>=1) room_dict = PyTuple_GET_ITEM(rp,0);

        if (room_dict && PyDict_Check(room_dict)){
            PyObject *key,*val; Py_ssize_t pos=0;
            while (PyDict_Next(room_dict, &pos, &key, &val)){
                PyObject* name = PyDict_GetItemString(val, "name");
                if (name && PyObject_RichCompareBool(name, room_name, Py_EQ)==1){
                    PyObject* outline = PyDict_GetItemString(val, "outline");
                    if (outline && PySequence_Check(outline) && PySequence_Size(outline)>0){
                        long minx=LONG_MAX, maxx=LONG_MIN, miny=LONG_MAX, maxy=LONG_MIN;
                        Py_ssize_t n = PySequence_Size(outline);
                        for (Py_ssize_t i=0;i<n;i++){
                            PyObject* pt = PySequence_GetItem(outline,i);
                            if (PyTuple_Check(pt) && PyTuple_GET_SIZE(pt)>=2){
                                long px = PyLong_AsLong(PyTuple_GetItem(pt,0));
                                long py = PyLong_AsLong(PyTuple_GetItem(pt,1));
                                if (px<minx) minx=px; if (px>maxx) maxx=px;
                                if (py<miny) miny=py; if (py>maxy) maxy=py;
                            }
                            Py_XDECREF(pt);
                        }
                        if (rand256){
                            minx = (long)llround((double)minx/10.0);
                            maxx = (long)llround((double)maxx/10.0);
                            miny = (long)llround((double)miny/10.0);
                            maxy = (long)llround((double)maxy/10.0);
                        }
                        return Py_BuildValue("(llll)", minx, maxx, miny, maxy);
                    }
                }
            }
        }
    }
    Py_XDECREF(rp);

    // fallback: rooms_pos
    PyObject* rooms_pos = PyObject_GetAttrString(self->handler, "rooms_pos");
    if (rooms_pos && PySequence_Check(rooms_pos)){
        Py_ssize_t n = PySequence_Size(rooms_pos);
        for (Py_ssize_t i=0;i<n;i++){
            PyObject* room = PySequence_GetItem(rooms_pos,i);
            PyObject* name = PyDict_GetItemString(room, "name");
            if (name && PyObject_RichCompareBool(name, room_name, Py_EQ)==1){
                PyObject* outline = PyDict_GetItemString(room, "outline");
                if (outline && PySequence_Check(outline) && PySequence_Size(outline)>0){
                    long minx=LONG_MAX, maxx=LONG_MIN, miny=LONG_MAX, maxy=LONG_MIN;
                    Py_ssize_t m = PySequence_Size(outline);
                    for (Py_ssize_t j=0;j<m;j++){
                        PyObject* pt = PySequence_GetItem(outline,j);
                        if (PyTuple_Check(pt) && PyTuple_GET_SIZE(pt)>=2){
                            long px = PyLong_AsLong(PyTuple_GetItem(pt,0));
                            long py = PyLong_AsLong(PyTuple_GetItem(pt,1));
                            if (px<minx) minx=px; if (px>maxx) maxx=px;
                            if (py<miny) miny=py; if (py>maxy) maxy=py;
                        }
                        Py_XDECREF(pt);
                    }
                    if (rand256){
                        minx = (long)llround((double)minx/10.0);
                        maxx = (long)llround((double)maxx/10.0);
                        miny = (long)llround((double)miny/10.0);
                        maxy = (long)llround((double)maxy/10.0);
                    }
                    Py_DECREF(room);
                    Py_XDECREF(rooms_pos);
                    return Py_BuildValue("(llll)", minx, maxx, miny, maxy);
                }
            }
            Py_XDECREF(room);
        }
    }
    Py_XDECREF(rooms_pos);

    Py_RETURN_NONE;
}

/* async_check_if_zoom_is_on(image_array, margin_size=100, zoom=False, rand256=False) -> ndarray */
static PyObject* AutoCrop_async_check_if_zoom_is_on(AutoCropObject* self, PyObject* args, PyObject* kw){
    PyObject* img_obj; int margin_size=100, zoom=0, rand256=0;
    static char* kwlist[] = {"image_array","margin_size","zoom","rand256", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iii", kwlist, &img_obj, &margin_size, &zoom, &rand256))
        return NULL;

    PyArrayObject* img_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &img_arr, img_obj) != 0) return NULL;

    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    PyObject* vstate = shared ? PyObject_GetAttrString(shared, "vacuum_state") : NULL;
    PyObject* zoomflag = shared ? PyObject_GetAttrString(shared, "image_auto_zoom") : NULL;

    int state_cleaning = 0;
    if (vstate && PyUnicode_Check(vstate)){
        PyObject* s = PyUnicode_FromString("cleaning");
        if (s){
            state_cleaning = PyObject_RichCompareBool(vstate, s, Py_EQ)==1;
            Py_DECREF(s);
        }
    }
    int allow_zoom = zoom && zoomflag && PyObject_IsTrue(zoomflag);

    int x0=0,y0=0,x1=W,y1=H;

    if (allow_zoom && state_cleaning){
        // current_room
        PyObject* robot_pos = PyObject_GetAttrString(self->handler, "robot_pos");
        PyObject* current_room = NULL;
        if (robot_pos && PyDict_Check(robot_pos)){
            current_room = PyDict_GetItemString(robot_pos, "in_room");
            if (current_room) Py_INCREF(current_room);
        }

        if (!current_room && rand256){
            PyObject* rpos = PyObject_GetAttrString(self->handler, "robot_position");
            if (rpos && PyTuple_Check(rpos) && PyTuple_GET_SIZE(rpos)>=2){
                long rx = PyLong_AsLong(PyTuple_GET_ITEM(rpos,0));
                long ry = PyLong_AsLong(PyTuple_GET_ITEM(rpos,1));
                int zoom_size = 800;
                x0 = (int)clampl(rx - zoom_size/2, 0, W);
                x1 = (int)clampl(rx + zoom_size/2, 0, W);
                y0 = (int)clampl(ry - zoom_size/2, 0, H);
                y1 = (int)clampl(ry + zoom_size/2, 0, H);
                Py_XDECREF(rpos);
                Py_XDECREF(robot_pos);
                Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
                // FIX: Call crop_rgba BEFORE decref
                PyObject* result = crop_rgba(img_arr, x0,y0,x1,y1);
                Py_DECREF(img_arr);
                return result;
            }
            Py_XDECREF(rpos);
        }

        if (current_room){
            // try async_get_room_bounding_box
            PyObject* kw2 = Py_BuildValue("{s:O,s:i}", "room_name", current_room, "rand256", rand256);
            PyObject* empty_args = Py_BuildValue("()");
            PyObject* bbox = AutoCrop_async_get_room_bounding_box(self, empty_args, kw2);
            Py_DECREF(empty_args);
            Py_DECREF(kw2);
            if (bbox && bbox != Py_None){
                long left, right, up, down;
                if (PyArg_ParseTuple(bbox, "llll", &left, &right, &up, &down)){
                    long tl = left - margin_size;
                    long tr = right + margin_size;
                    long tu = up - margin_size;
                    long td = down + margin_size;

                    x0 = (int)clampl(tl,0,W); x1=(int)clampl(tr,0,W);
                    y0 = (int)clampl(tu,0,H); y1=(int)clampl(td,0,H);
                    if (x1-x0<1 || y1-y0<1){ x0=0; y0=0; x1=W; y1=H; }
                    Py_DECREF(bbox);
                    Py_XDECREF(current_room); Py_XDECREF(robot_pos);
                    Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
                    // FIX: Call crop_rgba BEFORE decref
                    PyObject* result = crop_rgba(img_arr, x0,y0,x1,y1);
                    Py_DECREF(img_arr);
                    return result;
                }
                Py_DECREF(bbox);
            }
            Py_XDECREF(current_room);
        }

        // fallback to auto_crop if available
        if (self->auto_crop && self->auto_crop != Py_None && PyList_Check(self->auto_crop)){
            long l = PyLong_AsLong(PyList_GetItem(self->auto_crop,0));
            long u = PyLong_AsLong(PyList_GetItem(self->auto_crop,1));
            long r = PyLong_AsLong(PyList_GetItem(self->auto_crop,2));
            long d = PyLong_AsLong(PyList_GetItem(self->auto_crop,3));
            x0=(int)clampl(l,0,W); x1=(int)clampl(r,0,W);
            y0=(int)clampl(u,0,H); y1=(int)clampl(d,0,H);
            Py_XDECREF(robot_pos);
            Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
            // FIX: Call crop_rgba BEFORE decref
            PyObject* result = crop_rgba(img_arr, x0,y0,x1,y1);
            Py_DECREF(img_arr);
            return result;
        }
    }

    // Fallback: apply auto_crop when zoom is off so we still resize by detect_colour
    if (self->auto_crop && self->auto_crop != Py_None && PyList_Check(self->auto_crop)){
        long l = PyLong_AsLong(PyList_GetItem(self->auto_crop,0));
        long u = PyLong_AsLong(PyList_GetItem(self->auto_crop,1));
        long r = PyLong_AsLong(PyList_GetItem(self->auto_crop,2));
        long d = PyLong_AsLong(PyList_GetItem(self->auto_crop,3));
        x0=(int)clampl(l,0,W); x1=(int)clampl(r,0,W);
        y0=(int)clampl(u,0,H); y1=(int)clampl(d,0,H);
        Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
        // FIX: Call crop_rgba BEFORE decref
        PyObject* result = crop_rgba(img_arr, x0,y0,x1,y1);
        Py_DECREF(img_arr);
        return result;
    }

    // default: full copy
    Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
    // FIX: Create output and copy BEFORE decref img_arr (p points to img_arr data)
    npy_intp dims[3] = { H, W, 4 };
    PyObject* out = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!out) {
        Py_DECREF(img_arr);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)out), p, (size_t)H*(size_t)W*4);
    Py_DECREF(img_arr);
    return out;
}

/* async_rotate_the_image(trimmed, rotate) -> ndarray */
static PyObject* AutoCrop_async_rotate_the_image(AutoCropObject* self, PyObject* args){
    PyObject* arr; int rotate=0;
    if (!PyArg_ParseTuple(args, "Oi", &arr, &rotate)) return NULL;

    PyArrayObject* src_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &src_arr, arr) != 0) return NULL;

    PyObject* out = rot_rgba(src_arr, rotate);
    Py_DECREF(src_arr);

    // update crop_area same as Python
    if (rotate == 90 || rotate == 270){
        Py_XDECREF(self->crop_area);
        self->crop_area = make_list4(self->trim_left, self->trim_up, self->trim_right, self->trim_down);
    } else if (rotate == 180){
        Py_XDECREF(self->crop_area);
        Py_INCREF(self->auto_crop);
        self->crop_area = self->auto_crop;
    } else {
        Py_XDECREF(self->crop_area);
        Py_INCREF(self->auto_crop);
        self->crop_area = self->auto_crop;
    }
    return out;
}

/* auto_trim_and_zoom_image(...) -> ndarray */
static PyObject* AutoCrop_auto_trim_and_zoom_image(AutoCropObject* self, PyObject* args, PyObject* kw){
    PyObject* img_obj; PyObject* colour_obj=NULL;
    int margin_size=0, rotate=0, zoom=0, rand256=0;
    static char* kwlist[] = {"image_array","detect_colour","margin_size","rotate","zoom","rand256", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Oiiii", kwlist,
        &img_obj, &colour_obj, &margin_size, &rotate, &zoom, &rand256))
        return NULL;
    int need_decref_colour = 0;
    if (!colour_obj){ colour_obj = Py_BuildValue("(iiii)", 93,109,126,255); need_decref_colour = 1; }

    // self.auto_crop = await self._init_auto_crop()
    PyObject* ac = AutoCrop__init_auto_crop(self, NULL);
    Py_XDECREF(ac);

    if (self->auto_crop == Py_None || (PyList_Check(self->auto_crop) &&
        PyLong_AsLong(PyList_GetItem(self->auto_crop,0))==0 &&
        PyLong_AsLong(PyList_GetItem(self->auto_crop,1))==0 &&
        PyLong_AsLong(PyList_GetItem(self->auto_crop,2))==0 &&
        PyLong_AsLong(PyList_GetItem(self->auto_crop,3))==0))
    {
        // compute bbox via async_image_margins
        PyObject* margins_args = Py_BuildValue("(OO)", img_obj, colour_obj);
        PyObject* margins = AutoCrop_async_image_margins(self, margins_args);
        Py_DECREF(margins_args);
        if (!margins){ if (need_decref_colour) Py_DECREF(colour_obj); return NULL; }

        long min_y, min_x, max_x, max_y;
        if (!PyArg_ParseTuple(margins, "llll", &min_y, &min_x, &max_x, &max_y)){
            Py_DECREF(margins);
            if (need_decref_colour) Py_DECREF(colour_obj);
            return NULL;
        }
        Py_DECREF(margins);

        self->trim_left = (long)min_x - margin_size;
        self->trim_up   = (long)min_y - margin_size;
        self->trim_right= (long)max_x + margin_size;
        self->trim_down = (long)max_y + margin_size;

        // (trimmed_width, trimmed_height)
        PyObject* wh = AutoCrop__calculate_trimmed_dimensions(self, NULL);
        if (!wh){ if (need_decref_colour) Py_DECREF(colour_obj); return NULL; }
        long tw, th;
        PyArg_ParseTuple(wh, "ll", &tw, &th);
        Py_DECREF(wh);

        // check_trim(...) raise TrimError on failure
        PyObject* ct_args = Py_BuildValue("(lllOl)", th, tw, margin_size, img_obj, rotate);
        PyObject* ct = AutoCrop_check_trim(self, ct_args);
        Py_DECREF(ct_args);
        if (!ct){
            // exception set; let caller catch TrimError, but your Python path returns e.image
            // Here mimic your try/except: return the image (no crop)
            PyErr_Clear();
            if (need_decref_colour) Py_DECREF(colour_obj);
            Py_INCREF(img_obj);
            return img_obj;
        }
        Py_XDECREF(ct);

        // store auto_crop
        Py_XDECREF(self->auto_crop);
        self->auto_crop = make_list4(self->trim_left, self->trim_up, self->trim_right, self->trim_down);

        // best-effort sync to handler.crop_area as Python does
        if (self->handler && self->auto_crop){
            PyObject_SetAttrString(self->handler, "crop_area", self->auto_crop);
        }

        // update shared.trims (best-effort)
        PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
        if (shared){
            PyObject* trims_cls = PyObject_GetAttrString(self->handler, "shared"); // placeholder if you wire TrimsData.from_dict
            Py_XDECREF(trims_cls);
            Py_DECREF(shared);
        }

        (void)AutoCrop_auto_crop_offset(self, NULL);
        // after offset, keep handler.crop_area in sync
        if (self->handler && self->auto_crop){
            PyObject_SetAttrString(self->handler, "crop_area", self->auto_crop);
        }
    }

    // trimmed = await async_check_if_zoom_is_on(...)
    PyObject* args2 = Py_BuildValue("(O)", img_obj);
    PyObject* kw2   = Py_BuildValue("{s:i,s:i,s:i}", "margin_size", margin_size, "zoom", zoom, "rand256", rand256);
    PyObject* trimmed = AutoCrop_async_check_if_zoom_is_on(self, args2, kw2);
    Py_DECREF(args2); Py_DECREF(kw2);
    if (!trimmed){ if (need_decref_colour) Py_DECREF(colour_obj); return NULL; }

    // rotated = await async_rotate_the_image(trimmed, rotate)
    PyObject* rotate_args = Py_BuildValue("(Oi)", trimmed, rotate);
    PyObject* rotated = AutoCrop_async_rotate_the_image(self, rotate_args);
    Py_DECREF(rotate_args);
    Py_DECREF(trimmed);
    if (!rotated){ if (need_decref_colour) Py_DECREF(colour_obj); return NULL; }

    // self.handler.crop_img_size = [rotated.shape[1], rotated.shape[0]]
    PyArrayObject* rarr = (PyArrayObject*)rotated;
    int RH = (int)PyArray_DIM(rarr,0), RW=(int)PyArray_DIM(rarr,1);
    PyObject* size_list = Py_BuildValue("[ii]", RW, RH);
    PyObject_SetAttrString(self->handler, "crop_img_size", size_list);
    Py_DECREF(size_list);

    // Ensure handler.crop_area reflects the final (post-rotation) crop area
    if (self->handler && self->crop_area){
        PyObject_SetAttrString(self->handler, "crop_area", self->crop_area);
    }

    if (need_decref_colour) Py_DECREF(colour_obj);
    return rotated;
}

/* ----------------------- method table & type ----------------------- */

static PyMethodDef AutoCrop_methods[] = {
    {"validate_crop_dimensions", (PyCFunction)AutoCrop_validate_crop_dimensions, METH_VARARGS,
     "validate_crop_dimensions(shared) -> bool"},
    {"check_trim", (PyCFunction)AutoCrop_check_trim, METH_VARARGS,
     "check_trim(trimmed_height, trimmed_width, margin_size, image_array, file_name, rotate)"},
    {"_calculate_trimmed_dimensions", (PyCFunction)AutoCrop__calculate_trimmed_dimensions, METH_NOARGS,
     "_calculate_trimmed_dimensions() -> (trimmed_width, trimmed_height)"},
    {"_async_auto_crop_data", (PyCFunction)AutoCrop__async_auto_crop_data, METH_VARARGS,
     "_async_auto_crop_data(tdata) -> list|None"},
    {"auto_crop_offset", (PyCFunction)AutoCrop_auto_crop_offset, METH_NOARGS,
     "auto_crop_offset()"},
    {"_init_auto_crop", (PyCFunction)AutoCrop__init_auto_crop, METH_NOARGS,
     "_init_auto_crop() -> list|None"},
    {"async_image_margins", (PyCFunction)AutoCrop_async_image_margins, METH_VARARGS,
     "async_image_margins(image_array, detect_colour) -> (min_y, min_x, max_x, max_y)"},
    {"async_get_room_bounding_box", (PyCFunction)AutoCrop_async_get_room_bounding_box, METH_VARARGS|METH_KEYWORDS,
     "async_get_room_bounding_box(room_name, rand256=False) -> tuple|None"},
    {"async_check_if_zoom_is_on", (PyCFunction)AutoCrop_async_check_if_zoom_is_on, METH_VARARGS|METH_KEYWORDS,
     "async_check_if_zoom_is_on(image_array, margin_size=100, zoom=False, rand256=False) -> ndarray"},
    {"async_rotate_the_image", (PyCFunction)AutoCrop_async_rotate_the_image, METH_VARARGS,
     "async_rotate_the_image(trimmed, rotate) -> ndarray"},
    {"auto_trim_and_zoom_image", (PyCFunction)AutoCrop_auto_trim_and_zoom_image, METH_VARARGS|METH_KEYWORDS,
     "auto_trim_and_zoom_image(image_array, detect_colour=(93,109,126,255), margin_size=0, rotate=0, zoom=False, rand256=False) -> ndarray"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject AutoCropType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mvcrender.autocrop.AutoCrop",
    .tp_basicsize = sizeof(AutoCropObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc   = "C-backed AutoCrop (1:1 API with Python)",
    .tp_new   = PyType_GenericNew,
    .tp_init  = (initproc)AutoCrop_init,
    .tp_dealloc = (destructor)AutoCrop_dealloc,
    .tp_methods = AutoCrop_methods,
};

/* ------------------------------- module ------------------------------- */

static PyMethodDef module_methods[] = { {NULL, NULL, 0, NULL} };

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "AutoCrop native module",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__native(void){
    import_array();
    // Create TrimError exception
    TrimError = PyErr_NewException("mvcrender.autocrop.TrimError", NULL, NULL);
    if (!TrimError) return NULL;

    if (PyType_Ready(&AutoCropType) < 0) return NULL;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&AutoCropType);
    if (PyModule_AddObject(m, "AutoCrop", (PyObject*)&AutoCropType) != 0){
        Py_DECREF(&AutoCropType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(TrimError);
    if (PyModule_AddObject(m, "TrimError", TrimError) != 0){
        Py_DECREF(TrimError);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
