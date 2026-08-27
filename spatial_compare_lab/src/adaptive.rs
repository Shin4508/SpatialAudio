pub struct AdaptiveParams {
    pub early_gain: f32,
    pub late_send: f32,
    pub dry_anchor: f32,
}

pub struct AdaptiveController {
    early: f32,
    late: f32,
    dry: f32,
}

impl AdaptiveController {
    pub fn new() -> Self {
        Self { early: 0.85, late: 0.10, dry: 0.04 }
    }

    pub fn analyze(&mut self, left: &[f32], right: &[f32]) -> AdaptiveParams {
        let n = left.len().min(right.len()).max(1);
        let mut e_l = 0.0_f32;
        let mut e_r = 0.0_f32;
        let mut cross = 0.0_f32;
        let mut e_mid = 0.0_f32;
        let mut e_side = 0.0_f32;

        for i in 0..left.len().min(right.len()) {
            let l = left[i];
            let r = right[i];
            let m = 0.5*(l+r);
            let s = 0.5*(l-r);
            e_l += l*l;
            e_r += r*r;
            cross += l*r;
            e_mid += m*m;
            e_side += s*s;
        }

        let corr = if e_l > 1e-12 && e_r > 1e-12 {
            (cross / (e_l.sqrt()*e_r.sqrt())).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let coherence = corr.abs();
        let side_ratio = e_side / (e_mid + e_side + 1e-9);

        // Coherent center content: keep it focused and dry.
        // Diffuse/wide content: allow more room and a small dry anchor.
        let target_early = 0.72 + 0.38*(1.0 - coherence) + 0.12*side_ratio;
        let target_late = 0.06 + 0.14*(1.0 - coherence) + 0.08*side_ratio;
        let target_dry = 0.02 + 0.08*(1.0 - coherence) + 0.04*side_ratio;

        let smooth = 0.08;
        self.early += (target_early - self.early)*smooth;
        self.late += (target_late - self.late)*smooth;
        self.dry += (target_dry - self.dry)*smooth;

        let _ = n;
        AdaptiveParams {
            early_gain: self.early.clamp(0.55, 1.25),
            late_send: self.late.clamp(0.04, 0.24),
            dry_anchor: self.dry.clamp(0.0, 0.14),
        }
    }
}
