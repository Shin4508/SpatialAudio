use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use spatial_compare_lab_v2::{
    audio, ensure_rate,
    headphone_eq::HeadphoneEqConfig,
    hrtf::HrtfProfile,
    render_with_config,
    room::{EnvironmentKind, RoomConfig},
    RenderOptions,
};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::Semaphore;

#[derive(Clone)]
struct App {
    profile: Arc<HrtfProfile>,
    slots: Arc<Semaphore>,
}
#[derive(Serialize)]
struct Preset {
    id: &'static str,
    name: &'static str,
    width: f32,
    depth: f32,
    height: f32,
    distance: f32,
    rt60: f32,
    predelay: f32,
    reverb: f32,
}
#[derive(Deserialize, Debug)]
struct Parameters {
    environment: String,
    distance: f32,
    angle: f32,
    width: f32,
    depth: f32,
    height: f32,
    rt60: f32,
    predelay: f32,
    reverb: f32,
    adaptive: bool,
}
type ApiError = (StatusCode, String);
fn bad(e: impl std::fmt::Display) -> ApiError {
    (StatusCode::BAD_REQUEST, e.to_string())
}
impl Parameters {
    fn config(&self) -> Result<RoomConfig, ApiError> {
        for (name, value, min, max) in [
            ("distance", self.distance, 0.15, 10.0),
            ("angle", self.angle, 0.0, 90.0),
            ("width", self.width, 3.0, 80.0),
            ("depth", self.depth, 3.0, 100.0),
            ("height", self.height, 2.0, 40.0),
            ("RT60", self.rt60, 0.1, 5.0),
            ("predelay", self.predelay, 0.0, 100.0),
            ("reverb gain", self.reverb, 0.0, 1.0),
        ] {
            if !value.is_finite() || !(min..=max).contains(&value) {
                return Err(bad(format!("{name} must be between {min} and {max}")));
            }
        }
        let env = EnvironmentKind::all()
            .into_iter()
            .find(|e| e.slug() == self.environment)
            .ok_or_else(|| bad("Unknown environment"))?;
        let mut cfg = RoomConfig::preset(env);
        cfg.listener_x *= self.width / cfg.width_m;
        cfg.listener_y *= self.depth / cfg.depth_m;
        cfg.width_m = self.width;
        cfg.depth_m = self.depth;
        cfg.height_m = self.height;
        cfg.speaker_distance_m = self.distance;
        cfg.left_speaker_az_deg = -self.angle;
        cfg.right_speaker_az_deg = self.angle;
        cfg.set_web_reverb(self.rt60, self.predelay, self.reverb);
        cfg.validate().map_err(bad)?;
        Ok(cfg)
    }
}
async fn presets(State(app): State<App>) -> Json<serde_json::Value> {
    let presets: Vec<_> = EnvironmentKind::all()
        .into_iter()
        .map(|e| {
            let c = RoomConfig::preset(e);
            Preset {
                id: e.slug(),
                name: e.label(),
                width: c.width_m,
                depth: c.depth_m,
                height: c.height_m,
                distance: c.speaker_distance_m,
                rt60: c.late_params().rt60_s.max(0.1),
                predelay: c.late_params().predelay_ms,
                reverb: c.late_params().output_gain,
            }
        })
        .collect();
    Json(serde_json::json!({"sample_rate": app.profile.sample_rate, "presets": presets}))
}
async fn render(
    State(app): State<App>,
    Query(params): Query<Parameters>,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let cfg = params.config()?;
    let permit = app.slots.clone().try_acquire_owned().map_err(|_| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "Another render is running. Try again shortly.".into(),
        )
    })?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let _permit = permit;
        let run = || -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let reader = hound::WavReader::new(std::io::Cursor::new(&body))?;
            let spec = reader.spec();
            if spec.channels != 2 || spec.sample_rate != app.profile.sample_rate || reader.duration() == 0 || reader.duration() > spec.sample_rate * 180 {
                return Err("Upload a nonempty stereo WAV matching the HRTF sample rate, up to 180 seconds.".into());
            }
            let dir = tempfile::tempdir()?;
            let input = dir.path().join("input.wav"); let output = dir.path().join("result.wav");
            std::fs::write(&input, &body)?;
            let (l,r,sr) = audio::load_stereo_wav(input.to_str().unwrap())?;
            if l.iter().chain(&r).any(|v| !v.is_finite() || v.abs() > 16.0) { return Err("Audio contains invalid or excessive sample values".into()); }
            ensure_rate(sr, &app.profile)?;
            let eq = HeadphoneEqConfig::load_or_flat("")?;
            let opt = RenderOptions { direct_hrtf_intensity:1.0, environment:Some(cfg.environment), headphone_eq:false, adaptive:params.adaptive, distance_m:Some(params.distance), angle_deg:params.angle };
            render_with_config(output.to_str().unwrap(), &l,&r,sr,&app.profile,&eq,opt,Some(cfg))?;
            let (mut l,mut r,sr)=audio::load_stereo_wav(output.to_str().unwrap())?;
            if l.iter().chain(&r).any(|v| !v.is_finite()) { return Err("Render produced invalid samples".into()); }
            audio::prevent_clipping(&mut l,&mut r);
            audio::write_stereo_wav(output.to_str().unwrap(),&l,&r,sr)?;
            Ok(std::fs::read(output)?)
        };
        run().map_err(|e| e.to_string())
    }).await.map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR,"Render worker failed".into()))?.map_err(bad)?;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=spatial-result.wav",
            ),
            (header::CACHE_CONTROL, "no-store"),
        ],
        bytes,
    ))
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = std::env::var_os("HRTF_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../spatial_compare_lab/hrtf_profiles/default")
        });
    let profile = HrtfProfile::load(profile.to_str().ok_or("Invalid profile path")?)
        .map_err(|e| format!("Cannot load HRTF profile {}: {e}", profile.display()))?;
    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../static/index.html")) }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css")],
                    include_str!("../static/style.css"),
                )
            }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../static/app.js"),
                )
            }),
        )
        .route("/api/presets", get(presets))
        .route("/api/render", post(render))
        .layer(DefaultBodyLimit::max(70 * 1024 * 1024))
        .with_state(App {
            profile: Arc::new(profile),
            slots: Arc::new(Semaphore::new(1)),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("Spatial Sound: http://127.0.0.1:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn params() -> Parameters {
        Parameters {
            environment: "jazz_club".into(),
            distance: 1.5,
            angle: 30.0,
            width: 12.0,
            depth: 18.0,
            height: 4.0,
            rt60: 1.0,
            predelay: 20.0,
            reverb: 0.2,
            adaptive: false,
        }
    }
    #[test]
    fn rejects_invalid_parameters() {
        let mut p = params();
        p.distance = f32::NAN;
        assert!(p.config().is_err());
        p = params();
        p.environment = "unknown".into();
        assert!(p.config().is_err());
        p = params();
        p.width = 3.0;
        p.angle = 90.0;
        p.distance = 10.0;
        assert!(p.config().is_err());
    }
    #[test]
    fn applies_geometry_and_decay() {
        let p = params();
        let c = p.config().unwrap();
        assert_eq!(c.width_m, 12.0);
        assert_eq!(c.speaker_distance_m, 1.5);
        assert_eq!(c.late_params().rt60_s, 1.0);
    }
}
