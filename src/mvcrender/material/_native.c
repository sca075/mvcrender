// src/mvcrender/material/_native.c
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>

/*
 * Fast C implementation for material pattern rendering (tiles, wood)
 * Optimized for performance with direct pixel manipulation
 */

/* ==================== Helper Functions ==================== */

static inline void draw_horizontal_line(uint8_t* data, int width, int height, 
                                       int y, int x0, int x1, 
                                       uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                       int thickness) {
    if (y < 0 || y >= height || x0 >= width || x1 < 0) return;
    
    int start_x = (x0 < 0) ? 0 : x0;
    int end_x = (x1 >= width) ? width - 1 : x1;
    
    for (int t = 0; t < thickness && (y + t) < height; t++) {
        int row_offset = (y + t) * width * 4;
        for (int x = start_x; x <= end_x; x++) {
            int idx = row_offset + x * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = a;
        }
    }
}

static inline void draw_vertical_line(uint8_t* data, int width, int height,
                                      int x, int y0, int y1,
                                      uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                      int thickness) {
    if (x < 0 || x >= width || y0 >= height || y1 < 0) return;
    
    int start_y = (y0 < 0) ? 0 : y0;
    int end_y = (y1 >= height) ? height - 1 : y1;
    
    for (int t = 0; t < thickness && (x + t) < width; t++) {
        for (int y = start_y; y <= end_y; y++) {
            int idx = (y * width + x + t) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = a;
        }
    }
}

static inline void draw_rect_outline(uint8_t* data, int width, int height,
                                     int x0, int y0, int x1, int y1,
                                     uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                     int thickness) {
    if (x1 <= x0 || y1 <= y0) return;
    
    // Top and bottom lines
    draw_horizontal_line(data, width, height, y0, x0, x1 - 1, r, g, b, a, thickness);
    draw_horizontal_line(data, width, height, y1 - 1, x0, x1 - 1, r, g, b, a, thickness);
    
    // Left and right lines
    draw_vertical_line(data, width, height, x0, y0, y1 - 1, r, g, b, a, thickness);
    draw_vertical_line(data, width, height, x1 - 1, y0, y1 - 1, r, g, b, a, thickness);
}

/* ==================== Material Pattern Generators ==================== */

static PyObject* generate_tile_pattern(PyObject* self, PyObject* args, PyObject* kwargs) {
    int cells = 4;
    int pixel_size = 5;
    int r = 40, g = 40, b = 40, a = 45;
    
    static char* kwlist[] = {"cells", "pixel_size", "r", "g", "b", "a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiii", kwlist,
                                     &cells, &pixel_size, &r, &g, &b, &a)) {
        return NULL;
    }
    
    int size = cells * pixel_size;
    int thickness = (pixel_size <= 7) ? 1 : 2;
    
    // Create output array
    npy_intp dims[3] = {size, size, 4};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    if (!output) return NULL;
    
    uint8_t* data = (uint8_t*)PyArray_DATA(output);
    
    // Draw horizontal line at top
    draw_horizontal_line(data, size, size, 0, 0, size - 1, r, g, b, a, thickness);
    
    // Draw vertical line at left
    draw_vertical_line(data, size, size, 0, 0, size - 1, r, g, b, a, thickness);
    
    return (PyObject*)output;
}

static PyObject* generate_wood_horizontal(PyObject* self, PyObject* args, PyObject* kwargs) {
    int cells = 36;
    int pixel_size = 5;
    int r = 40, g = 40, b = 40, a = 38;
    
    static char* kwlist[] = {"cells", "pixel_size", "r", "g", "b", "a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiii", kwlist,
                                     &cells, &pixel_size, &r, &g, &b, &a)) {
        return NULL;
    }
    
    int tile_px = cells * pixel_size;
    int thickness = (pixel_size <= 7) ? 1 : 2;
    
    // Plank dimensions
    int plank_h_cells = 3;
    int plank_w_cells = 24;
    int plank_h = plank_h_cells * pixel_size;
    int plank_w = plank_w_cells * pixel_size;
    
    // Create output array
    npy_intp dims[3] = {tile_px, tile_px, 4};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    if (!output) return NULL;
    
    uint8_t* data = (uint8_t*)PyArray_DATA(output);
    
    int rows = (tile_px / plank_h) + 1;
    int cols = (tile_px + plank_w - 1) / plank_w + 1;
    
    // Draw staggered planks
    for (int r_idx = 0; r_idx <= rows; r_idx++) {
        int y0 = r_idx * plank_h;
        int y1 = y0 + plank_h;
        int offset = (r_idx % 2 == 1) ? (plank_w / 2) : 0;
        
        for (int c_idx = 0; c_idx <= cols; c_idx++) {
            int x0 = c_idx * plank_w - offset;
            int x1 = x0 + plank_w;

            // Clip to tile bounds
            int cx0 = (x0 < 0) ? 0 : x0;
            int cy0 = (y0 < 0) ? 0 : y0;
            int cx1 = (x1 > tile_px) ? tile_px : x1;
            int cy1 = (y1 > tile_px) ? tile_px : y1;

            draw_rect_outline(data, tile_px, tile_px, cx0, cy0, cx1, cy1, r, g, b, a, thickness);
        }
    }

    return (PyObject*)output;
}

static PyObject* generate_wood_vertical(PyObject* self, PyObject* args, PyObject* kwargs) {
    int cells = 36;
    int pixel_size = 5;
    int r = 40, g = 40, b = 40, a = 38;

    static char* kwlist[] = {"cells", "pixel_size", "r", "g", "b", "a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiii", kwlist,
                                     &cells, &pixel_size, &r, &g, &b, &a)) {
        return NULL;
    }

    int tile_px = cells * pixel_size;
    int thickness = (pixel_size <= 7) ? 1 : 2;

    // Plank dimensions
    int plank_w_cells = 3;
    int plank_h_cells = 24;
    int plank_w = plank_w_cells * pixel_size;
    int plank_h = plank_h_cells * pixel_size;

    // Create output array
    npy_intp dims[3] = {tile_px, tile_px, 4};
    PyArrayObject* output = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    if (!output) return NULL;

    uint8_t* data = (uint8_t*)PyArray_DATA(output);

    int cols = (tile_px / plank_w) + 1;
    int rows = (tile_px + plank_h - 1) / plank_h + 1;

    // Draw staggered planks
    for (int c_idx = 0; c_idx <= cols; c_idx++) {
        int x0 = c_idx * plank_w;
        int x1 = x0 + plank_w;
        int offset = (c_idx % 2 == 1) ? (plank_h / 2) : 0;

        for (int r_idx = 0; r_idx <= rows; r_idx++) {
            int y0 = r_idx * plank_h - offset;
            int y1 = y0 + plank_h;

            // Clip to tile bounds
            int cx0 = (x0 < 0) ? 0 : x0;
            int cy0 = (y0 < 0) ? 0 : y0;
            int cx1 = (x1 > tile_px) ? tile_px : x1;
            int cy1 = (y1 > tile_px) ? tile_px : y1;

            draw_rect_outline(data, tile_px, tile_px, cx0, cy0, cx1, cy1, r, g, b, a, thickness);
        }
    }

    return (PyObject*)output;
}

/* ==================== Module Definition ==================== */

static PyMethodDef material_methods[] = {
    {"generate_tile_pattern", (PyCFunction)generate_tile_pattern,
     METH_VARARGS | METH_KEYWORDS,
     "Generate a tile pattern.\n\n"
     "Args:\n"
     "    cells: Number of cells in the pattern\n"
     "    pixel_size: Size of each pixel/cell\n"
     "    r, g, b, a: RGBA color values\n\n"
     "Returns:\n"
     "    RGBA numpy array with tile pattern"},

    {"generate_wood_horizontal", (PyCFunction)generate_wood_horizontal,
     METH_VARARGS | METH_KEYWORDS,
     "Generate horizontal wood plank pattern.\n\n"
     "Args:\n"
     "    cells: Number of cells in the pattern\n"
     "    pixel_size: Size of each pixel/cell\n"
     "    r, g, b, a: RGBA color values\n\n"
     "Returns:\n"
     "    RGBA numpy array with wood pattern"},

    {"generate_wood_vertical", (PyCFunction)generate_wood_vertical,
     METH_VARARGS | METH_KEYWORDS,
     "Generate vertical wood plank pattern.\n\n"
     "Args:\n"
     "    cells: Number of cells in the pattern\n"
     "    pixel_size: Size of each pixel/cell\n"
     "    r, g, b, a: RGBA color values\n\n"
     "Returns:\n"
     "    RGBA numpy array with wood pattern"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef material_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Fast C implementation for material pattern rendering",
    -1,
    material_methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&material_module);
}

