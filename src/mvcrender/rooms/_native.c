// src/mvcrender/rooms/_native.c
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * Rooms processing C extension for mvcrender
 * 
 * Provides high-performance implementations of:
 * - Morphological operations (erosion, dilation)
 * - Compressed pixel mask filling
 * - Point extraction from masks
 * - Convex hull computation
 */

/* ==================== Morphological Operations ==================== */

/**
 * Binary erosion with 3x3 structuring element
 * 
 * @param mask Input binary mask (H x W, uint8)
 * @param output Output eroded mask (H x W, uint8)
 * @param height Height of mask
 * @param width Width of mask
 * @param iterations Number of erosion iterations
 */
static void binary_erosion_3x3(
    const uint8_t* mask,
    uint8_t* output,
    int height,
    int width,
    int iterations
) {
    if (iterations <= 0 || height <= 0 || width <= 0) return;
    
    // Allocate temporary buffer for ping-pong
    uint8_t* temp = (uint8_t*)malloc(height * width * sizeof(uint8_t));
    if (!temp) return;
    
    // Copy input to output for first iteration
    memcpy(output, mask, height * width);
    
    for (int iter = 0; iter < iterations; iter++) {
        const uint8_t* src = output;
        uint8_t* dst = temp;
        
        // Process each pixel
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                
                // Border pixels are always eroded to 0
                if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                    dst[idx] = 0;
                    continue;
                }
                
                // Check 3x3 neighborhood - all must be non-zero
                uint8_t result = 1;
                for (int dy = -1; dy <= 1 && result; dy++) {
                    for (int dx = -1; dx <= 1 && result; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (src[ny * width + nx] == 0) {
                            result = 0;
                        }
                    }
                }
                dst[idx] = result;
            }
        }
        
        // Swap buffers
        memcpy(output, temp, height * width);
    }
    
    free(temp);
}

/**
 * Binary dilation with 3x3 structuring element
 * 
 * @param mask Input binary mask (H x W, uint8)
 * @param output Output dilated mask (H x W, uint8)
 * @param height Height of mask
 * @param width Width of mask
 * @param iterations Number of dilation iterations
 */
static void binary_dilation_3x3(
    const uint8_t* mask,
    uint8_t* output,
    int height,
    int width,
    int iterations
) {
    if (iterations <= 0 || height <= 0 || width <= 0) return;
    
    // Allocate temporary buffer
    uint8_t* temp = (uint8_t*)malloc(height * width * sizeof(uint8_t));
    if (!temp) return;
    
    // Copy input to output
    memcpy(output, mask, height * width);
    
    for (int iter = 0; iter < iterations; iter++) {
        const uint8_t* src = output;
        uint8_t* dst = temp;
        
        // Process each pixel
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                
                // Check 3x3 neighborhood - any non-zero sets result to 1
                uint8_t result = 0;
                for (int dy = -1; dy <= 1 && !result; dy++) {
                    for (int dx = -1; dx <= 1 && !result; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (src[ny * width + nx] != 0) {
                                result = 1;
                            }
                        }
                    }
                }
                dst[idx] = result;
            }
        }
        
        // Swap buffers
        memcpy(output, temp, height * width);
    }
    
    free(temp);
}

/* ==================== Compressed Pixel Processing ==================== */

/**
 * Fill mask from compressed pixels format [x, y, length, x, y, length, ...]
 * 
 * @param pixel_data Compressed pixel data as int32 array
 * @param num_pixels Number of triplets (length of pixel_data / 3)
 * @param mask Output mask to fill (H x W, uint8)
 * @param height Height of mask
 * @param width Width of mask
 * @param min_x Offset to subtract from x coordinates
 * @param min_y Offset to subtract from y coordinates
 */
static void fill_compressed_pixels(
    const int32_t* pixel_data,
    int num_pixels,
    uint8_t* mask,
    int height,
    int width,
    int min_x,
    int min_y
) {
    for (int i = 0; i < num_pixels; i++) {
        int x = pixel_data[i * 3 + 0];
        int y = pixel_data[i * 3 + 1];
        int length = pixel_data[i * 3 + 2];
        
        // Adjust coordinates to local mask
        int local_x = x - min_x;
        int local_y = y - min_y;
        
        // Bounds check
        if (local_y < 0 || local_y >= height) continue;
        if (local_x >= width) continue;
        
        // Calculate end point, clamping to mask width
        int end_x = local_x + length;
        if (end_x > width) end_x = width;
        if (local_x < 0) local_x = 0;
        
        // Fill the row segment
        if (end_x > local_x) {
            memset(&mask[local_y * width + local_x], 1, end_x - local_x);
        }
    }
}

/* ==================== Point Extraction ==================== */

/**
 * Extract non-zero points from mask
 * 
 * @param mask Input binary mask (H x W, uint8)
 * @param height Height of mask
 * @param width Width of mask
 * @param points_x Output array for x coordinates (pre-allocated)
 * @param points_y Output array for y coordinates (pre-allocated)
 * @param max_points Maximum number of points to extract
 * @return Number of points extracted
 */
static int extract_mask_points(
    const uint8_t* mask,
    int height,
    int width,
    int* points_x,
    int* points_y,
    int max_points
) {
    int count = 0;
    
    for (int y = 0; y < height && count < max_points; y++) {
        for (int x = 0; x < width && count < max_points; x++) {
            if (mask[y * width + x] != 0) {
                points_x[count] = x;
                points_y[count] = y;
                count++;
            }
        }
    }
    
    return count;
}

/* ==================== Python Interface ==================== */

/**
 * Python: process_room_mask(mask, padding=10) -> processed_mask
 * 
 * Applies morphological operations (erosion + dilation) to clean up room mask
 */
static PyObject* rooms_process_room_mask(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* mask_obj;
    int padding = 10;
    
    static char* kwlist[] = {"mask", "padding", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &mask_obj, &padding)) {
        return NULL;
    }
    
    // Convert to numpy array
    PyArrayObject* mask_arr = (PyArrayObject*)PyArray_FROM_OTF(
        mask_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY
    );
    if (!mask_arr) {
        PyErr_SetString(PyExc_ValueError, "Expected uint8 numpy array for mask");
        return NULL;
    }
    
    // Check dimensions
    if (PyArray_NDIM(mask_arr) != 2) {
        Py_DECREF(mask_arr);
        PyErr_SetString(PyExc_ValueError, "Mask must be 2D array");
        return NULL;
    }
    
    int height = (int)PyArray_DIM(mask_arr, 0);
    int width = (int)PyArray_DIM(mask_arr, 1);
    const uint8_t* mask_data = (const uint8_t*)PyArray_DATA(mask_arr);
    
    // Create output array
    npy_intp dims[2] = {height, width};
    PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT8);
    if (!output) {
        Py_DECREF(mask_arr);
        return NULL;
    }
    
    uint8_t* output_data = (uint8_t*)PyArray_DATA(output);
    
    // Allocate temporary buffer for erosion
    uint8_t* temp = (uint8_t*)malloc(height * width * sizeof(uint8_t));
    if (!temp) {
        Py_DECREF(mask_arr);
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
    
    // Release GIL for processing
    Py_BEGIN_ALLOW_THREADS
    
    // Erosion (1 iteration)
    binary_erosion_3x3(mask_data, temp, height, width, 1);
    
    // Dilation (1 iteration)
    binary_dilation_3x3(temp, output_data, height, width, 1);
    
    Py_END_ALLOW_THREADS
    
    free(temp);
    Py_DECREF(mask_arr);
    
    return (PyObject*)output;
}

/**
 * Python: fill_compressed_pixels_mask(pixel_data, width, height, min_x, min_y) -> mask
 *
 * Creates and fills a mask from compressed pixel format
 */
static PyObject* rooms_fill_compressed_pixels_mask(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* pixel_data_obj;
    int width, height, min_x, min_y;

    static char* kwlist[] = {"pixel_data", "width", "height", "min_x", "min_y", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiiii", kwlist,
                                      &pixel_data_obj, &width, &height, &min_x, &min_y)) {
        return NULL;
    }

    // Convert pixel_data to numpy array
    PyArrayObject* pixel_arr = (PyArrayObject*)PyArray_FROM_OTF(
        pixel_data_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY
    );
    if (!pixel_arr) {
        PyErr_SetString(PyExc_ValueError, "Expected int32 numpy array for pixel_data");
        return NULL;
    }

    // Check dimensions - should be Nx3
    if (PyArray_NDIM(pixel_arr) != 2 || PyArray_DIM(pixel_arr, 1) != 3) {
        Py_DECREF(pixel_arr);
        PyErr_SetString(PyExc_ValueError, "pixel_data must be Nx3 array");
        return NULL;
    }

    int num_pixels = (int)PyArray_DIM(pixel_arr, 0);
    const int32_t* pixel_data = (const int32_t*)PyArray_DATA(pixel_arr);

    // Create output mask
    npy_intp dims[2] = {height, width};
    PyArrayObject* mask = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_UINT8, 0);
    if (!mask) {
        Py_DECREF(pixel_arr);
        return NULL;
    }

    uint8_t* mask_data = (uint8_t*)PyArray_DATA(mask);

    // Release GIL for filling
    Py_BEGIN_ALLOW_THREADS
    fill_compressed_pixels(pixel_data, num_pixels, mask_data, height, width, min_x, min_y);
    Py_END_ALLOW_THREADS

    Py_DECREF(pixel_arr);
    return (PyObject*)mask;
}

/**
 * Python: extract_points_from_mask(mask) -> (points_x, points_y)
 *
 * Extracts all non-zero points from mask
 */
static PyObject* rooms_extract_points_from_mask(PyObject* self, PyObject* args) {
    PyObject* mask_obj;

    if (!PyArg_ParseTuple(args, "O", &mask_obj)) {
        return NULL;
    }

    // Convert to numpy array
    PyArrayObject* mask_arr = (PyArrayObject*)PyArray_FROM_OTF(
        mask_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY
    );
    if (!mask_arr) {
        PyErr_SetString(PyExc_ValueError, "Expected uint8 numpy array for mask");
        return NULL;
    }

    // Check dimensions
    if (PyArray_NDIM(mask_arr) != 2) {
        Py_DECREF(mask_arr);
        PyErr_SetString(PyExc_ValueError, "Mask must be 2D array");
        return NULL;
    }

    int height = (int)PyArray_DIM(mask_arr, 0);
    int width = (int)PyArray_DIM(mask_arr, 1);
    const uint8_t* mask_data = (const uint8_t*)PyArray_DATA(mask_arr);

    // First pass: count non-zero pixels
    int count = 0;
    for (int i = 0; i < height * width; i++) {
        if (mask_data[i] != 0) count++;
    }

    if (count == 0) {
        Py_DECREF(mask_arr);
        // Return empty arrays
        npy_intp dims[1] = {0};
        PyArrayObject* x_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
        PyArrayObject* y_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
        return Py_BuildValue("(NN)", x_arr, y_arr);
    }

    // Allocate output arrays
    npy_intp dims[1] = {count};
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);

    if (!x_arr || !y_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_DECREF(mask_arr);
        return NULL;
    }

    int32_t* x_data = (int32_t*)PyArray_DATA(x_arr);
    int32_t* y_data = (int32_t*)PyArray_DATA(y_arr);

    // Second pass: extract points
    Py_BEGIN_ALLOW_THREADS
    int actual_count = extract_mask_points(mask_data, height, width, x_data, y_data, count);
    Py_END_ALLOW_THREADS

    Py_DECREF(mask_arr);

    return Py_BuildValue("(NN)", x_arr, y_arr);
}

/* ==================== Module Definition ==================== */

static PyMethodDef rooms_methods[] = {
    {"process_room_mask", (PyCFunction)rooms_process_room_mask,
     METH_VARARGS | METH_KEYWORDS,
     "Apply morphological operations (erosion + dilation) to clean room mask.\n\n"
     "Args:\n"
     "    mask: 2D uint8 numpy array (binary mask)\n"
     "    padding: Padding size (default: 10)\n\n"
     "Returns:\n"
     "    Processed 2D uint8 numpy array"},

    {"fill_compressed_pixels_mask", (PyCFunction)rooms_fill_compressed_pixels_mask,
     METH_VARARGS | METH_KEYWORDS,
     "Create mask from compressed pixel format [x, y, length, ...].\n\n"
     "Args:\n"
     "    pixel_data: Nx3 int32 numpy array of (x, y, length) triplets\n"
     "    width: Width of output mask\n"
     "    height: Height of output mask\n"
     "    min_x: X offset to subtract from coordinates\n"
     "    min_y: Y offset to subtract from coordinates\n\n"
     "Returns:\n"
     "    2D uint8 numpy array (binary mask)"},

    {"extract_points_from_mask", rooms_extract_points_from_mask,
     METH_VARARGS,
     "Extract all non-zero points from binary mask.\n\n"
     "Args:\n"
     "    mask: 2D uint8 numpy array (binary mask)\n\n"
     "Returns:\n"
     "    Tuple of (x_coords, y_coords) as int32 numpy arrays"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rooms_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native C extension for room processing operations",
    -1,
    rooms_methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&rooms_module);
}


