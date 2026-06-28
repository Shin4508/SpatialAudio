use ndarray::Array1;

fn get_low_shelf_coeffs(sr: f32, f0: f32, gain_db: f32) {
    let pi: f32 = 3.141592653589793;
    let ten: f32 = 10.0;
    let a: f32 = ten.powf(gain_db / 40.0);
    let w0 = 2.0 * pi * f0 / sr;
}
