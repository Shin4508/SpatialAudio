use ndarray::Array1;

fn get_low_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: Option<f32>) -> ([f32; 3], [f32; 3]) {
    use std::f32::consts::PI;
    let q = q.unwrap_or(0.707);
    let a: f32 = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha: f32 = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0: f32 = w0.cos();
    let sqrt_a_alpha_2: f32 = 2.0 * a.sqrt() * alpha;

    let b0: f32 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1: f32 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2: f32 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0: f32 = (a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1: f32 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2: f32 = (a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2;

    ([b0, b1, b2], [a0, a1, a2])
}

fn get_high_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: Option<f32>) -> ([f32; 3], [f32; 3]) {
    use std::f32::consts::PI;
    let q = q.unwrap_or(0.707);
    let a: f32 = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha: f32 = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0: f32 = w0.cos();
    let sqrt_a_alpha_2: f32 = 2.0 * a.sqrt() * alpha;
    
    let b0: f32 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1: f32 = -2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2: f32 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0: f32 = (a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1: f32 = 2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2: f32 = (a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2;
    
    ([b0, b1, b2], [a0, a1, a2])
    
}

struct RealTimeBiquadFilter {
   b: Vec<f32>,
   a: Vec<f32>,
}

impl RealTimeBiquadFilter{
    fn new(b: Vec<f32>, a:Vec<f32>) -> Self{
        RealTimeBiquadFilter {b, a}
    }

    fn process(&self, block: i32){
        let output: 
    }
}
