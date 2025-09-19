#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*
 C-backed AutoCrop class:
 - Holds reference to Python `handler`
 - Keeps trims/offsets in C fields
 - Methods:
   * init_auto_crop()
   * image_margins(image_array, detect_colour) -> (min_y, min_x, max_x, max_y)
   * check_if_zoom_is_on(image_array, margin_size=100, zoom=False, rand256=False) -> trimmed (new ndarray)
   * rotate_image(trimmed, rotate) -> rotated (new ndarray)
   * auto_trim_and_zoom_image(image_array, detect_colour, margin_size, rotate, zoom, rand256) -> rotated
*/

typedef struct {
    PyObject_HEAD
    PyObject* handler;          // borrowed reference to your handler
    PyObject* auto_crop;        // list [l, u, r, d] or None
    PyObject* crop_area;        // list or None

    // trims
    long trim_up, trim_down, trim_left, trim_right;
    // offsets
    long offset_top, offset_bottom, offset_left, offset_right;
} AutoCropObject;

/* ---------------------------- Utilities ---------------------------- */

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

static inline long clampl(long v, long lo, long hi) {
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

/* Read handler.shared.trims.to_dict() and offsets */
static int load_trims_offsets(AutoCropObject* self) {
    // handler.shared.trims.to_dict()
    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    if (!shared) return -1;
    PyObject* trims = PyObject_GetAttrString(shared, "trims");
    if (!trims) { Py_DECREF(shared); return -1; }
    PyObject* to_dict = PyObject_GetAttrString(trims, "to_dict");
    if (!to_dict) { Py_DECREF(trims); Py_DECREF(shared); return -1; }
    PyObject* dict = PyObject_CallObject(to_dict, NULL);
    Py_DECREF(to_dict);
    Py_DECREF(trims);
    if (!dict) { Py_DECREF(shared); return -1; }

    long tu = 0, td = 0, tl = 0, tr = 0;
    PyObject* v;
    v = PyDict_GetItemString(dict, "trim_up");    if (v) tu = PyLong_AsLong(v);
    v = PyDict_GetItemString(dict, "trim_down");  if (v) td = PyLong_AsLong(v);
    v = PyDict_GetItemString(dict, "trim_left");  if (v) tl = PyLong_AsLong(v);
    v = PyDict_GetItemString(dict, "trim_right"); if (v) tr = PyLong_AsLong(v);
    Py_DECREF(dict);

    self->trim_up = tu; self->trim_down = td; self->trim_left = tl; self->trim_right = tr;

    // offsets
    PyObject* off_top    = PyObject_GetAttrString(shared, "offset_top");
    PyObject* off_down   = PyObject_GetAttrString(shared, "offset_down");
    PyObject* off_left   = PyObject_GetAttrString(shared, "offset_left");
    PyObject* off_right  = PyObject_GetAttrString(shared, "offset_right");
    self->offset_top    = off_top   ? PyLong_AsLong(off_top)   : 0;
    self->offset_bottom = off_down  ? PyLong_AsLong(off_down)  : 0;
    self->offset_left   = off_left  ? PyLong_AsLong(off_left)  : 0;
    self->offset_right  = off_right ? PyLong_AsLong(off_right) : 0;
    Py_XDECREF(off_top); Py_XDECREF(off_down); Py_XDECREF(off_left); Py_XDECREF(off_right);

    Py_DECREF(shared);
    return 0;
}

static void apply_auto_crop_offset(AutoCropObject* self) {
    if (!self->auto_crop || self->auto_crop == Py_None) return;
    PyObject* item0 = PyList_GetItem(self->auto_crop, 0);
    PyObject* item1 = PyList_GetItem(self->auto_crop, 1);
    PyObject* item2 = PyList_GetItem(self->auto_crop, 2);
    PyObject* item3 = PyList_GetItem(self->auto_crop, 3);
    long l = PyLong_AsLong(item0) + self->offset_left;
    long u = PyLong_AsLong(item1) + self->offset_top;
    long r = PyLong_AsLong(item2) - self->offset_right;
    long d = PyLong_AsLong(item3) - self->offset_bottom;
    Py_DECREF(self->auto_crop);
    self->auto_crop = make_list4(l,u,r,d);
}

/* Update shared.image_ref_width/height */
static void update_shared_ref_dims(AutoCropObject* self, long tw, long th) {
    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    if (!shared) return;
    PyObject_SetAttrString(shared, "image_ref_width", PyLong_FromLong(tw));
    PyObject_SetAttrString(shared, "image_ref_height", PyLong_FromLong(th));
    Py_DECREF(shared);
}

/* -------------------------- bbox helpers -------------------------- */

/* Foreground = alpha >= threshold */
static int bbox_from_alpha(uint8_t* rgba, int H, int W, int threshold, int* x0,int* y0,int* x1,int* y1) {
    int xmin=W, ymin=H, xmax=-1, ymax=-1;
    for (int y=0; y<H; y++) {
        uint8_t* row = rgba + ((size_t)y*(size_t)W)*4;
        for (int x=0; x<W; x++) {
            uint8_t a = row[(size_t)x*4 + 3];
            if (a >= threshold) {
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

/* Foreground = any channel != detect_colour (RGBA match) */
static int bbox_from_color(uint8_t* rgba, int H, int W, const uint8_t dc[4], int* x0,int* y0,int* x1,int* y1) {
    int xmin=W, ymin=H, xmax=-1, ymax=-1;
    for (int y=0; y<H; y++) {
        uint8_t* row = rgba + ((size_t)y*(size_t)W)*4;
        for (int x=0; x<W; x++) {
            size_t idx = (size_t)x*4;
            uint8_t r=row[idx], g=row[idx+1], b=row[idx+2], a=row[idx+3];
            if (!(r==dc[0] && g==dc[1] && b==dc[2] && a==dc[3])) {
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

/* -------------------------- ndarray ops -------------------------- */

/* Crop: returns a new H'xW'x4 array, copies data */
static PyObject* crop_rgba(PyArrayObject* src_arr, int x0,int y0,int x1,int y1) {
    int H = (int)PyArray_DIM(src_arr,0);
    int W = (int)PyArray_DIM(src_arr,1);
    int cw = x1 - x0;
    int ch = y1 - y0;
    if (cw <= 0 || ch <= 0) {
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

    for (int yy=0; yy<ch; yy++) {
        uint8_t* srow = src + ((size_t)(y0+yy) * src_stride) + (size_t)x0 * 4;
        uint8_t* drow = dst + ((size_t)yy * dst_stride);
        memcpy(drow, srow, dst_stride);
    }
    return out;
}

/* Rotate 90/180/270 into a new array */
static PyObject* rot_rgba(PyArrayObject* src_arr, int rotate) {
    int H = (int)PyArray_DIM(src_arr,0);
    int W = (int)PyArray_DIM(src_arr,1);
    uint8_t* src = (uint8_t*)PyArray_DATA(src_arr);

    npy_intp dims90[3] = { W, H, 4 };
    npy_intp dims00[3] = { H, W, 4 };
    PyObject* out = NULL;

    if (rotate == 0) {
        // Just return a copy
        out = PyArray_SimpleNew(3, dims00, NPY_UINT8);
        if (!out) return NULL;
        memcpy(PyArray_DATA((PyArrayObject*)out), src, (size_t)H*(size_t)W*4);
        return out;
    } else if (rotate == 90 || rotate == 270) {
        out = PyArray_SimpleNew(3, dims90, NPY_UINT8);
    } else if (rotate == 180) {
        out = PyArray_SimpleNew(3, dims00, NPY_UINT8);
    } else {
        PyErr_SetString(PyExc_ValueError, "rotate must be 0, 90, 180, or 270");
        return NULL;
    }
    if (!out) return NULL;

    uint8_t* dst = (uint8_t*)PyArray_DATA((PyArrayObject*)out);

    if (rotate == 180) {
        for (int y=0; y<H; y++) {
            for (int x=0; x<W; x++) {
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                size_t di = ((size_t)(H-1-y) * (size_t)W + (size_t)(W-1-x)) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
    } else if (rotate == 90) {
        int OH=W, OW=H;
        for (int y=0; y<H; y++) {
            for (int x=0; x<W; x++) {
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                int ny = x;
                int nx = OW - 1 - y;
                size_t di = ((size_t)ny * (size_t)OW + (size_t)nx) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
    } else { // 270
        int OH=W, OW=H;
        for (int y=0; y<H; y++) {
            for (int x=0; x<W; x++) {
                size_t si = ((size_t)y * (size_t)W + (size_t)x) * 4;
                int ny = OH - 1 - x;
                int nx = y;
                size_t di = ((size_t)ny * (size_t)OW + (size_t)nx) * 4;
                dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; dst[di+3]=src[si+3];
            }
        }
    }
    return out;
}

/* -------------------------- AutoCrop type -------------------------- */

static int AutoCrop_init(AutoCropObject* self, PyObject* args, PyObject* kw) {
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

static void AutoCrop_dealloc(AutoCropObject* self) {
    Py_XDECREF(self->handler);
    Py_XDECREF(self->auto_crop);
    Py_XDECREF(self->crop_area);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init_auto_crop(self) -> list|None */
static PyObject* AutoCrop_init_auto_crop(AutoCropObject* self, PyObject* Py_UNUSED(ignored)) {
    // If we already have auto_crop, keep behavior similar to Python version
    if (self->auto_crop != Py_None && PyList_Check(self->auto_crop)) {
        // handler.max_frames = 1205
        PyObject* v = PyLong_FromLong(1205);
        if (v) {
            PyObject_SetAttrString(self->handler, "max_frames", v);
            Py_DECREF(v);
        }
        Py_INCREF(self->auto_crop);
        return self->auto_crop;
    }

    // Build trims list [l,u,r,d]; if all zero -> return None
    if (self->trim_left || self->trim_up || self->trim_right || self->trim_down) {
        Py_XDECREF(self->auto_crop);
        self->auto_crop = make_list4(self->trim_left, self->trim_up, self->trim_right, self->trim_down);
        apply_auto_crop_offset(self);
        Py_INCREF(self->auto_crop);
        return self->auto_crop;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/* image_margins(self, image_array, detect_colour(tuple4)) -> (min_y, min_x, max_x, max_y) */
static PyObject* AutoCrop_image_margins(AutoCropObject* self, PyObject* args) {
    PyObject* img_obj; PyObject* colour_obj;
    if (!PyArg_ParseTuple(args, "OO", &img_obj, &colour_obj)) return NULL;

    PyArrayObject* img_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &img_arr, img_obj) != 0) return NULL;

    uint8_t dc[4] = {93,109,126,255}; // default if caller gave wrong thing
    if (PySequence_Check(colour_obj) && PySequence_Size(colour_obj) == 4) {
        for (int i=0;i<4;i++) {
            PyObject* it = PySequence_GetItem(colour_obj, i);
            dc[i] = (uint8_t)PyLong_AsLong(it);
            Py_XDECREF(it);
        }
    }

    int x0,y0,x1,y1, ok=0;
    if (dc[3] == 0) ok = bbox_from_alpha(p,H,W,1,&x0,&y0,&x1,&y1);
    else           ok = bbox_from_color(p,H,W,dc,&x0,&y0,&x1,&y1);

    Py_DECREF(img_arr);

    if (!ok) {
        // no foreground, return full image as a safe fallback
        return Py_BuildValue("(iiii)", 0, 0, W-1, H-1);
    }
    // match your original return order (min_y, min_x, max_x, max_y)
    return Py_BuildValue("(iiii)", y0, x0, x1, y1);
}

/* check_if_zoom_is_on(self, image_array, margin_size=100, zoom=False, rand256=False) -> trimmed ndarray */
static PyObject* AutoCrop_check_if_zoom_is_on(AutoCropObject* self, PyObject* args, PyObject* kw) {
    PyObject* img_obj; int margin_size=100; int zoom=0; int rand256=0;
    static char* kwlist[] = {"image_array","margin_size","zoom","rand256", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iii", kwlist, &img_obj, &margin_size, &zoom, &rand256))
        return NULL;

    PyArrayObject* img_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &img_arr, img_obj) != 0) return NULL;

    PyObject* shared = PyObject_GetAttrString(self->handler, "shared");
    PyObject* vstate = shared ? PyObject_GetAttrString(shared, "vacuum_state") : NULL;
    PyObject* zoomflag = shared ? PyObject_GetAttrString(shared, "image_auto_zoom") : NULL;
    int state_cleaning = (vstate && PyUnicode_Check(vstate) && PyObject_RichCompareBool(vstate, PyUnicode_FromString("cleaning"), Py_EQ)==1);
    int allow_zoom = zoom && zoomflag && PyObject_IsTrue(zoomflag);

    int x0=0,y0=0,x1=W,y1=H;

    if (allow_zoom && state_cleaning) {
        // handler.robot_pos.get("in_room") if available
        PyObject* robot_pos = PyObject_GetAttrString(self->handler, "robot_pos");
        PyObject* current_room = NULL;
        if (robot_pos && PyDict_Check(robot_pos)) {
            current_room = PyDict_GetItemString(robot_pos, "in_room");
            if (current_room) Py_INCREF(current_room);
        }

        if (!current_room && rand256) {
            // try robot_position
            PyObject* rpos = PyObject_GetAttrString(self->handler, "robot_position");
            if (rpos && PyTuple_Check(rpos) && PyTuple_GET_SIZE(rpos)>=2) {
                long rx = PyLong_AsLong(PyTuple_GET_ITEM(rpos,0));
                long ry = PyLong_AsLong(PyTuple_GET_ITEM(rpos,1));
                int zoom_size = 800;
                x0 = (int)clAMPL(rx - zoom_size/2, 0, W);
                x1 = (int)clAMPL(rx + zoom_size/2, 0, W);
                y0 = (int)clAMPL(ry - zoom_size/2, 0, H);
                y1 = (int)clAMPL(ry + zoom_size/2, 0, H);
                Py_XDECREF(rpos);
                Py_XDECREF(robot_pos);
                Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
                Py_DECREF(img_arr);
                return crop_rgba(img_arr, x0,y0,x1,y1);
            }
            Py_XDECREF(rpos);
        }

        if (current_room) {
            // compute bbox from room outline: try handler.room_propriety then rooms_pos
            int found = 0;
            // room_propriety
            PyObject* rp = PyObject_GetAttrString(self->handler, "room_propriety");
            if (rp && rp != Py_None) {
                PyObject* room_dict = NULL;
                if (PyDict_Check(rp)) room_dict = rp;
                else if (PyTuple_Check(rp) && PyTuple_GET_SIZE(rp)>=1) room_dict = PyTuple_GET_ITEM(rp,0);

                if (room_dict && PyDict_Check(room_dict)) {
                    PyObject *key, *val;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(room_dict, &pos, &key, &val)) {
                        PyObject* name = PyDict_GetItemString(val, "name");
                        if (name && PyObject_RichCompareBool(name, current_room, Py_EQ)==1) {
                            PyObject* outline = PyDict_GetItemString(val, "outline");
                            if (outline && PySequence_Check(outline) && PySequence_Size(outline)>0) {
                                long minx=LONG_MAX, maxx=LONG_MIN, miny=LONG_MAX, maxy=LONG_MIN;
                                Py_ssize_t n = PySequence_Size(outline);
                                for (Py_ssize_t i=0;i<n;i++){
                                    PyObject* pt = PySequence_GetItem(outline,i);
                                    if (PyTuple_Check(pt) && PyTuple_GET_SIZE(pt)>=2) {
                                        long px = PyLong_AsLong(PyTuple_GetItem(pt,0));
                                        long py = PyLong_AsLong(PyTuple_GetItem(pt,1));
                                        if (px<minx) minx=px; if (px>maxx) maxx=px;
                                        if (py<miny) miny=py; if (py>maxy) maxy=py;
                                    }
                                    Py_XDECREF(pt);
                                }
                                x0 = (int)(minx - margin_size);
                                x1 = (int)(maxx + margin_size);
                                y0 = (int)(miny - margin_size);
                                y1 = (int)(maxy + margin_size);
                                found = 1;
                            }
                        }
                    }
                }
            }
            Py_XDECREF(rp);

            if (!found) {
                PyObject* rooms_pos = PyObject_GetAttrString(self->handler, "rooms_pos");
                if (rooms_pos && PySequence_Check(rooms_pos)) {
                    Py_ssize_t n = PySequence_Size(rooms_pos);
                    for (Py_ssize_t i=0;i<n && !found;i++){
                        PyObject* room = PySequence_GetItem(rooms_pos,i);
                        PyObject* name = PyDict_GetItemString(room, "name");
                        if (name && PyObject_RichCompareBool(name, current_room, Py_EQ)==1) {
                            PyObject* outline = PyDict_GetItemString(room, "outline");
                            if (outline && PySequence_Check(outline) && PySequence_Size(outline)>0) {
                                long minx=LONG_MAX, maxx=LONG_MIN, miny=LONG_MAX, maxy=LONG_MIN;
                                Py_ssize_t m = PySequence_Size(outline);
                                for (Py_ssize_t j=0;j<m;j++){
                                    PyObject* pt = PySequence_GetItem(outline,j);
                                    if (PyTuple_Check(pt) && PyTuple_GET_SIZE(pt)>=2) {
                                        long px = PyLong_AsLong(PyTuple_GetItem(pt,0));
                                        long py = PyLong_AsLong(PyTuple_GetItem(pt,1));
                                        if (px<minx) minx=px; if (px>maxx) maxx=px;
                                        if (py<miny) miny=py; if (py>maxy) maxy=py;
                                    }
                                    Py_XDECREF(pt);
                                }
                                x0 = (int)(minx - margin_size);
                                x1 = (int)(maxx + margin_size);
                                y0 = (int)(miny - margin_size);
                                y1 = (int)(maxy + margin_size);
                                found = 1;
                            }
                        }
                        Py_XDECREF(room);
                    }
                }
                Py_XDECREF(rooms_pos);
            }

            if (found) {
                x0 = (int)clampf(x0, 0, W); x1 = (int)clampf(x1, 0, W);
                y0 = (int)clampf(y0, 0, H); y1 = (int)clampf(y1, 0, H);
                if (x1 - x0 < 1 || y1 - y0 < 1) { x0=0; y0=0; x1=W; y1=H; }
                Py_XDECREF(current_room);
                Py_XDECREF(robot_pos);
                Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
                Py_DECREF(img_arr);
                return crop_rgba(img_arr, x0,y0,x1,y1);
            }
            Py_XDECREF(current_room);
        }

        // Fallback to precomputed auto_crop if present
        if (self->auto_crop && self->auto_crop != Py_None && PyList_Check(self->auto_crop)) {
            long l = PyLong_AsLong(PyList_GetItem(self->auto_crop,0));
            long u = PyLong_AsLong(PyList_GetItem(self->auto_crop,1));
            long r = PyLong_AsLong(PyList_GetItem(self->auto_crop,2));
            long d = PyLong_AsLong(PyList_GetItem(self->auto_crop,3));
            x0=(int)clampl(l,0,W); x1=(int)clampl(r,0,W);
            y0=(int)clampl(u,0,H); y1=(int)clampl(d,0,H);
            Py_XDECREF(robot_pos);
            Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
            Py_DECREF(img_arr);
            return crop_rgba(img_arr, x0,y0,x1,y1);
        }
    }

    // Default: return full image copy
    Py_XDECREF(vstate); Py_XDECREF(zoomflag); Py_XDECREF(shared);
    Py_DECREF(img_arr);

    npy_intp dims[3] = { H, W, 4 };
    PyObject* out = PyArray_SimpleNew(3, dims, NPY_UINT8);
    if (!out) return NULL;
    memcpy(PyArray_DATA((PyArrayObject*)out), p, (size_t)H*(size_t)W*4);
    return out;
}

/* rotate_image(self, trimmed, rotate) -> rotated */
static PyObject* AutoCrop_rotate_image(AutoCropObject* self, PyObject* args) {
    PyObject* arr; int rotate=0;
    if (!PyArg_ParseTuple(args, "Oi", &arr, &rotate)) return NULL;
    PyArrayObject* src_arr; uint8_t* p; int H,W;
    if (as_rgba_nd(&p, &H, &W, &src_arr, arr) != 0) return NULL;

    PyObject* out = rot_rgba(src_arr, rotate);
    Py_DECREF(src_arr);

    // Update crop_area attribute similar to your Python logic
    if (self->crop_area) { Py_DECREF(self->crop_area); self->crop_area = Py_None; Py_INCREF(Py_None); }
    // (We keep it simple: store [trim_left, trim_up, trim_right, trim_down])
    self->crop_area = make_list4(self->trim_left, self->trim_up, self->trim_right, self->trim_down);
    if (!self->crop_area) { Py_DECREF(out); return NULL; }

    return out;
}

/* init_auto_trim_and_zoom_image(...) -> rotated */
static PyObject* AutoCrop_auto_trim_and_zoom_image(AutoCropObject* self, PyObject* args, PyObject* kw) {
    PyObject* img_obj; PyObject* colour_obj = NULL;
    int margin_size = 0, rotate = 0, zoom = 0, rand256 = 0;
    static char* kwlist[] = {"image_array","detect_colour","margin_size","rotate","zoom","rand256", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iiii", kwlist,
        &img_obj, &colour_obj, &margin_size, &rotate, &zoom, &rand256))
        return NULL;

    // Ensure auto_crop state is initialized/adjusted
    PyObject* ac = AutoCrop_init_auto_crop(self, NULL);
    Py_XDECREF(ac);

    // If auto_crop empty, calculate bbox from colour/alpha to set trims & shared dims
    if (self->auto_crop == Py_None) {
        PyArrayObject* img_arr; uint8_t* p; int H,W;
        if (as_rgba_nd(&p, &H, &W, &img_arr, img_obj) != 0) return NULL;

        uint8_t dc[4] = {93,109,126,255};
        if (colour_obj && PySequence_Check(colour_obj) && PySequence_Size(colour_obj)==4) {
            for (int i=0;i<4;i++){ PyObject* it=PySequence_GetItem(colour_obj,i); dc[i]=(uint8_t)PyLong_AsLong(it); Py_XDECREF(it); }
        }

        int x0,y0,x1,y1,ok=0;
        if (dc[3]==0) ok = bbox_from_alpha(p,H,W,1,&x0,&y0,&x1,&y1);
        else          ok = bbox_from_color(p,H,W,dc,&x0,&y0,&x1,&y1);

        if (!ok) { x0=0; y0=0; x1=W-1; y1=H-1; }

        long l = (long)x0 - margin_size;
        long u = (long)y0 - margin_size;
        long r = (long)x1 + margin_size;
        long d = (long)y1 + margin_size;

        self->trim_left=l; self->trim_up=u; self->trim_right=r; self->trim_down=d;

        long tw = (r - self->offset_right) - (l + self->offset_left);
        long th = (d - self->offset_bottom) - (u + self->offset_top);
        if (tw < 1) tw = 1; if (th < 1) th = 1;
        update_shared_ref_dims(self, tw, th);

        Py_DECREF(img_arr);

        // store and apply offsets
        Py_XDECREF(self->auto_crop);
        self->auto_crop = make_list4(l,u,r,d);
        apply_auto_crop_offset(self);
    }

    // Zoom or crop-to-auto-crop
    PyObject* trimmed = AutoCrop_check_if_zoom_is_on(self,
        Py_BuildValue("(Oiiii)", img_obj, margin_size, zoom, rand256), NULL);
    if (!trimmed) return NULL;

    // Free input memory early
    // (Python GC will handle; here just proceed.)

    // Rotate
    PyObject* rotated = AutoCrop_rotate_image(self, Py_BuildValue("(Oi)", trimmed, rotate));
    Py_DECREF(trimmed);
    if (!rotated) return NULL;

    // Update handler.crop_img_size = [W,H]
    PyArrayObject* rarr = (PyArrayObject*)rotated;
    int RH = (int)PyArray_DIM(rarr,0), RW=(int)PyArray_DIM(rarr,1);
    PyObject* size_list = Py_BuildValue("[ii]", RW, RH);
    PyObject_SetAttrString(self->handler, "crop_img_size", size_list);
    Py_DECREF(size_list);

    return rotated;
}

/* ----------------------- Methods table & type ----------------------- */

static PyMethodDef AutoCrop_methods[] = {
    {"init_auto_crop", (PyCFunction)AutoCrop_init_auto_crop, METH_NOARGS, "Initialize auto-crop data and apply offsets."},
    {"image_margins", (PyCFunction)AutoCrop_image_margins, METH_VARARGS, "Find bbox of non-background pixels."},
    {"check_if_zoom_is_on", (PyCFunction)AutoCrop_check_if_zoom_is_on, METH_VARARGS|METH_KEYWORDS, "Zoom around room/robot or use auto-crop."},
    {"rotate_image", (PyCFunction)AutoCrop_rotate_image, METH_VARARGS, "Rotate RGBA image by 0/90/180/270."},
    {"auto_trim_and_zoom_image", (PyCFunction)AutoCrop_auto_trim_and_zoom_image, METH_VARARGS|METH_KEYWORDS, "Full pipeline: init, bbox, zoom, rotate."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject AutoCropType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mvcrender_autocrop.AutoCrop",
    .tp_basicsize = sizeof(AutoCropObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc   = "C-backed AutoCrop",
    .tp_new   = PyType_GenericNew,
    .tp_init  = (initproc)AutoCrop_init,
    .tp_dealloc = (destructor)AutoCrop_dealloc,
    .tp_methods = AutoCrop_methods,
};

/* ----------------------------- Module init ----------------------------- */

static PyMethodDef module_methods[] = { {NULL, NULL, 0, NULL} };

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "AutoCrop native module",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    if (PyType_Ready(&AutoCropType) < 0) return NULL;
    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&AutoCropType);
    if (PyModule_AddObject(m, "AutoCrop", (PyObject*)&AutoCropType) != 0) {
        Py_DECREF(&AutoCropType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
