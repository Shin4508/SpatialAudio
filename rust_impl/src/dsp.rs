use nalgebra::{Matrix2, Vector2};
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
    pub b: [f32; 3],
    pub a: [f32; 3],
    pub zi: [f32; 2],
}

impl RealTimeBiquadFilter {
    pub fn new(b: [f32; 3], a: [f32; 3]) -> Self {
        let mut filter = Self {
            b,
            a,
            zi: [0.0, 0.0],
        };
        filter.zi = filter.lfilter_zi();
        filter
    }

    fn lfilter_zi(&self) -> [f32; 2] {
        let a0 = self.a[0];
        let b1 = self.b[1] / a0;
        let b2 = self.b[2] / a0;
        let a1 = self.a[1] / a0;
        let a2 = self.a[2] / a0;

        let m_a = Matrix2::new(1.0 + a1, -1.0, a2, 1.0);

        let b0 = self.b[0] / a0;
        let v_b = Vector2::new(b1 - b0 * a1, b2 - b0 * a2);

        match m_a.lu().solve(&v_b) {
            Some(solution) => [solution[0], solution[1]],
            None => [0.0, 0.0],
        }
    }

    fn process(&mut self, block: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(block.len());
        let a0 = self.a[0];

        for &x in block {
            let y = (self.b[0] * x + self.zi[0]) / a0;
            self.zi[0] = self.b[1] * x + -self.a[1] * y + self.zi[1];
            self.zi[1] = self.b[2] * x - self.a[2] * y;
            output.push(y);
        }
        output
    }
}
