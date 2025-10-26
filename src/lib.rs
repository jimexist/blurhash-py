use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::f32::consts::PI;

/// Clamp a value between 0 and 255 and cast to u8
fn clamp_to_ubyte(src: i32) -> u8 {
    src.clamp(0, 255) as u8
}

// Use lazy_static for global cache of sRGB to linear conversion

lazy_static::lazy_static! {
    static ref SRGB_TO_LINEAR_CACHE: [f32; 256] = {
        let mut arr = [0.0f32; 256];
        for x in 0..256 {
            arr[x] = srgb_to_linear(x as u8);
        }
        arr
    };
}

/// Get sRGB to linear value from the cache (thread-safe, initializes on first use).
fn srgb_to_linear_cached(value: u8) -> f32 {
    SRGB_TO_LINEAR_CACHE[value as usize]
}

/// sRGB to linear, float in [0,1]
fn srgb_to_linear(value: u8) -> f32 {
    let v = value as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// linear to sRGB, u8 in [0,255]
fn linear_to_srgb(value: f32) -> i32 {
    let v = value.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        (v * 12.92 * 255.0 + 0.5) as i32
    } else {
        ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0 + 0.5) as i32
    }
}

fn sign_pow(value: f32, exp: f32) -> f32 {
    value.abs().powf(exp).copysign(value)
}

// Charset as slice (does not need to become a Vec every time)
const CHARSET: &[u8; 83] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

fn decode_to_int(string: &str, start: usize, end: usize) -> Option<i32> {
    let mut value = 0i32;
    let bytes = string.as_bytes();
    for idx in start..end {
        let &b = bytes.get(idx)?;
        let index = CHARSET.iter().position(|&c| c == b)? as i32;
        value = value * 83 + index;
    }
    Some(value)
}

fn encode_int(mut value: i32, length: usize) -> String {
    let mut chars = vec![0u8; length];
    for i in (0..length).rev() {
        chars[i] = CHARSET[(value % 83) as usize];
        value /= 83;
    }
    // Avoid unwrap(), map error properly for idiomatic code but here panic is fine
    // since the charset is correct
    String::from_utf8(chars).expect("Charset only contains valid UTF-8")
}

fn encode_dc(r: f32, g: f32, b: f32) -> i32 {
    let [rounded_r, rounded_g, rounded_b] =
        [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)];
    (rounded_r << 16) + (rounded_g << 8) + rounded_b
}

fn encode_ac(r: f32, g: f32, b: f32, maximum_value: f32) -> i32 {
    fn quant(v: f32, max: f32) -> i32 {
        (sign_pow(v / max, 0.5) * 9.0 + 9.5)
            .floor()
            .clamp(0.0, 18.0) as i32
    }
    let quant_r = quant(r, maximum_value);
    let quant_g = quant(g, maximum_value);
    let quant_b = quant(b, maximum_value);

    quant_r * 19 * 19 + quant_g * 19 + quant_b
}

/// Computes cosines for given n, bound, and scale.
/// Returns a Vec<f32> of cos(PI * n * i / scale) for each i in 0..bound.
fn precompute_cosines(n: i32, bound: i32, scale: i32) -> Vec<f32> {
    let mut cosines = Vec::with_capacity(bound as usize);
    let n = n as f32;
    let scale = scale as f32;
    for i in 0..bound {
        let val = (PI * n * i as f32 / scale).cos();
        cosines.push(val);
    }
    cosines
}

fn multiply_basis_function(
    x_component: i32,
    y_component: i32,
    width: i32,
    height: i32,
    rgb: &[u8],
    bytes_per_row: usize,
    cos_x: &[f32], // NEW: Precomputed cosines for x
    cos_y: &[f32], // NEW: Precomputed cosines for y
) -> (f32, f32, f32) {
    let mut result = [0.0f32; 3];
    let normalisation = if x_component == 0 && y_component == 0 {
        1.0
    } else {
        2.0
    };

    for y in 0..height {
        for x in 0..width {
            let basis = cos_x[x as usize] * cos_y[y as usize];
            let base = 3 * x as usize + y as usize * bytes_per_row;
            let channels = [rgb[base], rgb[base + 1], rgb[base + 2]];
            let linear = [
                srgb_to_linear_cached(channels[0]),
                srgb_to_linear_cached(channels[1]),
                srgb_to_linear_cached(channels[2]),
            ];
            for (acc, v) in result.iter_mut().zip(linear) {
                *acc += basis * v;
            }
        }
    }
    let scale = normalisation / (width * height) as f32;
    (result[0] * scale, result[1] * scale, result[2] * scale)
}

pub fn blurhash_for_pixels(
    x_components: i32,
    y_components: i32,
    width: i32,
    height: i32,
    rgb: &[u8],
    bytes_per_row: usize,
) -> Option<String> {
    if !(1..=9).contains(&x_components) || !(1..=9).contains(&y_components) {
        return None;
    }
    let num_factors = (x_components * y_components) as usize;
    let mut factors: Vec<[f32; 3]> = vec![[0.0; 3]; num_factors];

    // Precompute all cos_x for each x_component
    let mut all_cos_x: Vec<Vec<f32>> = Vec::with_capacity(x_components as usize);
    for x in 0..x_components {
        all_cos_x.push(precompute_cosines(x, width, width));
    }

    // Precompute all cos_y for each y_component
    let mut all_cos_y: Vec<Vec<f32>> = Vec::with_capacity(y_components as usize);
    for y in 0..y_components {
        all_cos_y.push(precompute_cosines(y, height, height));
    }

    for y in 0..y_components {
        for x in 0..x_components {
            let cos_x = &all_cos_x[x as usize];
            let cos_y = &all_cos_y[y as usize];
            let (r, g, b) =
                multiply_basis_function(x, y, width, height, rgb, bytes_per_row, cos_x, cos_y);
            factors[(y * x_components + x) as usize] = [r, g, b];
        }
    }

    let dc = factors[0];
    let ac = &factors[1..];

    // size flag
    let size_flag = (x_components - 1) + (y_components - 1) * 9;
    let mut ret = encode_int(size_flag, 1);

    // compute maximum value for AC components
    let maximum_value = ac
        .iter()
        .flat_map(|rgb| rgb.iter().map(|c| c.abs()))
        .fold(0.0f32, f32::max);

    let quant_max_value = if !ac.is_empty() {
        ((maximum_value * 166.0 - 0.5).floor().clamp(0.0, 82.0)) as i32
    } else {
        0
    };
    let norm_max_value = if !ac.is_empty() {
        (quant_max_value as f32 + 1.0) / 166.0
    } else {
        1.0
    };

    ret += &encode_int(quant_max_value, 1);

    // encode DC
    ret += &encode_int(encode_dc(dc[0], dc[1], dc[2]), 4);

    // encode AC
    for &acv in ac {
        ret += &encode_int(encode_ac(acv[0], acv[1], acv[2], norm_max_value), 2);
    }
    Some(ret)
}

fn decode_dc(value: i32) -> [f32; 3] {
    let r = ((value >> 16) & 0xFF) as u8;
    let g = ((value >> 8) & 0xFF) as u8;
    let b = (value & 0xFF) as u8;
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)]
}

fn decode_ac(value: i32, maximum_value: f32) -> [f32; 3] {
    let quant_r = value / (19 * 19);
    let quant_g = (value / 19) % 19;
    let quant_b = value % 19;
    [
        sign_pow((quant_r as f32 - 9.0) / 9.0, 2.0) * maximum_value,
        sign_pow((quant_g as f32 - 9.0) / 9.0, 2.0) * maximum_value,
        sign_pow((quant_b as f32 - 9.0) / 9.0, 2.0) * maximum_value,
    ]
}

fn decode_blurhash(blurhash: &str, width: usize, height: usize, punch: f32) -> Option<Vec<u8>> {
    let blurhash = blurhash.trim();
    if blurhash.len() < 6 {
        return None;
    }

    let size_flag = decode_to_int(blurhash, 0, 1)? as usize;
    let num_y = (size_flag / 9) + 1;
    let num_x = (size_flag % 9) + 1;

    let expected_length = 4 + 2 * num_x * num_y;
    if blurhash.len() != expected_length {
        return None;
    }

    let quant_max_value = decode_to_int(blurhash, 1, 2)? as f32;
    let max_ac = if quant_max_value > 0.0 {
        (quant_max_value + 1.0) / 166.0
    } else {
        1.0
    };

    let dc_value = decode_to_int(blurhash, 2, 6)?;
    let mut factors = Vec::with_capacity(num_x * num_y);
    factors.push(decode_dc(dc_value));

    for i in 0..(num_x * num_y - 1) {
        let ac_value = decode_to_int(blurhash, 6 + 2 * i, 6 + 2 * i + 2)?;
        factors.push(decode_ac(ac_value, max_ac * punch));
    }

    // Precompute cosines for all basis function indices and all pixel positions for efficiency
    let mut cosines_x = Vec::with_capacity(num_x);
    for i in 0..num_x {
        cosines_x.push(precompute_cosines(i as i32, width as i32, width as i32));
    }
    let mut cosines_y = Vec::with_capacity(num_y);
    for j in 0..num_y {
        cosines_y.push(precompute_cosines(j as i32, height as i32, height as i32));
    }

    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
            for j in 0..num_y {
                for i in 0..num_x {
                    let basis = cosines_x[i][x] * cosines_y[j][y];
                    let [fr, fg, fb] = factors[j * num_x + i];
                    r += fr * basis;
                    g += fg * basis;
                    b += fb * basis;
                }
            }
            let idx = 3 * (x + y * width);
            pixels[idx] = clamp_to_ubyte(linear_to_srgb(r));
            pixels[idx + 1] = clamp_to_ubyte(linear_to_srgb(g));
            pixels[idx + 2] = clamp_to_ubyte(linear_to_srgb(b));
        }
    }
    Some(pixels)
}

#[pyfunction]
fn decode_blurhash_py(
    py: Python<'_>,
    blurhash: &str,
    width: usize,
    height: usize,
    punch: Option<f32>,
) -> PyResult<Py<PyBytes>> {
    let punch = punch.unwrap_or(1.0);
    decode_blurhash(blurhash, width, height, punch)
        .map(|bytes| PyBytes::new(py, &bytes).into())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Failed to decode blurhash"))
}

#[pyfunction]
fn blurhash_for_pixels_py(
    x_components: usize,
    y_components: usize,
    width: usize,
    height: usize,
    rgb: Vec<u8>,
    bytes_per_row: usize,
) -> PyResult<Option<String>> {
    Ok(blurhash_for_pixels(
        x_components as i32,
        y_components as i32,
        width as i32,
        height as i32,
        &rgb,
        bytes_per_row,
    ))
}

/// Checks if the given blurhash string is potentially valid.
pub fn is_valid_blurhash(blurhash: &str) -> bool {
    if blurhash.len() < 6 {
        return false;
    }

    let size_flag = match decode_to_int(blurhash, 0, 1) {
        Some(val) => val as usize,
        None => return false,
    };

    let num_y = (size_flag / 9) + 1;
    let num_x = (size_flag % 9) + 1;

    blurhash.len() == 4 + 2 * num_x * num_y
}

#[pyfunction]
fn is_valid_blurhash_py(blurhash: &str) -> PyResult<bool> {
    Ok(is_valid_blurhash(blurhash))
}

#[pymodule]
#[pyo3(name = "_lib_name")]
fn blurhash_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blurhash_for_pixels_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_blurhash_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_blurhash_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_blurhash() {
        assert!(is_valid_blurhash("UPK^Kft0_K.9=FxWI.bw^JxuS$NGD,V@xtt7"));
        assert!(!is_valid_blurhash(""));
    }

    #[test]
    fn test_decode_blurhash() {
        let hash = "LlMF%n00%#MwS|WCWEM{R*bbWBbH";
        let width = 416;
        let height = 416;
        let decoded = decode_blurhash(hash, width, height, 1.0);
        assert!(decoded.is_some());
        let img = decoded.unwrap();
        assert_eq!(img.len(), width * height * 3);
    }

    #[test]
    fn test_decode_punch() {
        // This test checks if decoding with punch=2 returns a valid image buffer of expected length.
        let hash = "LlMF%n00%#MwS|WCWEM{R*bbWBbH";
        let width = 416;
        let height = 416;
        let punch = 2.0;
        let decoded = decode_blurhash(hash, width, height, punch);
        assert!(decoded.is_some());
        let img = decoded.unwrap();
        assert_eq!(img.len(), width * height * 3);
    }

    #[test]
    fn test_decode_invalid_blurhash() {
        // An invalid (too short) blurhash string should return None
        let hash = "#MwS|WCWEM{R";
        let width = 416;
        let height = 416;
        let decoded = decode_blurhash(hash, width, height, 1.0);
        assert!(decoded.is_none());
    }
}
