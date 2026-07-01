// C++ port of the classical SH-WFS inner loop: thresholded centre-of-gravity
// centroiding + per-sub-aperture slope computation.
//
// This is a faithful, allocation-free reimplementation of
// methods/common/centroid.py :: centroid_frame followed by the slope step in
// methods/modal_zernike/pipeline.py :: slopes_from_frame. It is the piece the
// README earmarks for a C port ("centroiding + reconstruction inner loop"),
// and the dominant per-frame cost of the classical/modal method.
//
// Everything is double precision and the pixel traversal is row-major (r outer,
// c inner) to match numpy's flattening, so results agree with the reference to
// floating-point round-off. No dependency beyond the C++ standard library.
//
// Layout / conventions (identical to the Python reference):
//   * image is (n, n) row-major; sub-aperture (i, j) occupies rows
//     [i*ppl, (i+1)*ppl) and cols [j*ppl, (j+1)*ppl).
//   * per window: thr = threshold_frac * max(window); w = max(pixel - thr, 0);
//     centroid = origin + sum(w * local_coord) / sum(w), where the local pixel
//     centre coordinate is (index + 0.5). If sum(w) <= 0 the centroid falls
//     back to the window centre.
//   * slopes are (centroid - reference) * slope_scale, returned as the
//     concatenation [sx over all valid sub-apertures, then sy] to match
//     np.concatenate([sx, sy]).

#include <cstddef>

extern "C" {

// cx_out, cy_out : length n_valid   (absolute detector-pixel centroids)
// slopes_out     : length 2*n_valid ([sx..., sy...])
void centroid_slopes(
    const double* image, int n, int ppl,
    const int* valid_i, const int* valid_j, int n_valid,
    const double* ref_x, const double* ref_y,
    double threshold_frac, double slope_scale,
    double* cx_out, double* cy_out, double* slopes_out)
{
    const double half = ppl / 2.0;

    for (int k = 0; k < n_valid; ++k) {
        const int y0 = valid_i[k] * ppl;   // top row of this sub-aperture window
        const int x0 = valid_j[k] * ppl;   // left column

        // Pass 1: window maximum (matches numpy win.max()).
        double wmax = image[(std::size_t)y0 * n + x0];
        for (int r = 0; r < ppl; ++r) {
            const double* row = image + (std::size_t)(y0 + r) * n + x0;
            for (int c = 0; c < ppl; ++c) {
                if (row[c] > wmax) wmax = row[c];
            }
        }
        const double thr = threshold_frac * wmax;

        // Pass 2: thresholded weighted centre-of-gravity.
        double total = 0.0, mx = 0.0, my = 0.0;
        for (int r = 0; r < ppl; ++r) {
            const double* row = image + (std::size_t)(y0 + r) * n + x0;
            const double ly = r + 0.5;                 // local y (row) coordinate
            for (int c = 0; c < ppl; ++c) {
                double wv = row[c] - thr;
                if (wv < 0.0) wv = 0.0;                 // clip(win - thr, 0, None)
                total += wv;
                mx += wv * (c + 0.5);                   // local x (col) coordinate
                my += wv * ly;
            }
        }

        double cx, cy;
        if (total <= 0.0) {                             // no signal -> window centre
            cx = x0 + half;
            cy = y0 + half;
        } else {
            cx = x0 + mx / total;
            cy = y0 + my / total;
        }

        cx_out[k] = cx;
        cy_out[k] = cy;
        slopes_out[k] = (cx - ref_x[k]) * slope_scale;              // sx block
        slopes_out[n_valid + k] = (cy - ref_y[k]) * slope_scale;    // sy block
    }
}

}  // extern "C"
