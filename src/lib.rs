use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::f32::consts::PI;

/// Clamp a value between 0 and 255 and cast to u8
fn clamp_to_ubyte(src: i32) -> u8 {
    if src >= 0 && src <= 255 {
        src as u8
    } else if src < 0 {
        0
    } else {
        255
    }
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
    if v <= 0.0031308 {
        (v * 12.92 * 255.0 + 0.5) as i32
    } else {
        ((1.055 * v.powf(1.0 / 2.4) - 0.055) * 255.0 + 0.5) as i32
    }
}

fn sign_pow(value: f32, exp: f32) -> f32 {
    value.abs().powf(exp).copysign(value)
}

const CHARSET: &[u8; 83] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

fn decode_to_int(string: &str, start: usize, end: usize) -> Option<i32> {
    let chars: Vec<u8> = CHARSET.to_vec();
    let mut value: i32 = 0;
    let bytes = string.as_bytes();
    for idx in start..end {
        let b = bytes.get(idx)?;
        let index = chars.iter().position(|&c| c == *b)? as i32;
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
    String::from_utf8(chars).unwrap()
}

fn encode_dc(r: f32, g: f32, b: f32) -> i32 {
    let rounded_r = linear_to_srgb(r);
    let rounded_g = linear_to_srgb(g);
    let rounded_b = linear_to_srgb(b);
    (rounded_r << 16) + (rounded_g << 8) + rounded_b
}

fn encode_ac(r: f32, g: f32, b: f32, maximum_value: f32) -> i32 {
    let quant_r = (sign_pow(r / maximum_value, 0.5) * 9.0 + 9.5)
        .floor()
        .clamp(0.0, 18.0) as i32;
    let quant_g = (sign_pow(g / maximum_value, 0.5) * 9.0 + 9.5)
        .floor()
        .clamp(0.0, 18.0) as i32;
    let quant_b = (sign_pow(b / maximum_value, 0.5) * 9.0 + 9.5)
        .floor()
        .clamp(0.0, 18.0) as i32;

    quant_r * 19 * 19 + quant_g * 19 + quant_b
}

/// Multiply basis function, taking (x,y) component, src width/height, source pixels, and bytes_per_row.
/// src is a flat &[u8] slice, RGB pixels, row-major, for each row (bytes_per_row wide).
fn multiply_basis_function(
    x_component: i32,
    y_component: i32,
    width: i32,
    height: i32,
    rgb: &[u8],
    bytes_per_row: usize,
) -> (f32, f32, f32) {
    let mut result = (0.0f32, 0.0f32, 0.0f32);
    let normalisation = if x_component == 0 && y_component == 0 {
        1.0
    } else {
        2.0
    };

    for y in 0..height {
        for x in 0..width {
            let basis = (PI * x_component as f32 * x as f32 / width as f32).cos()
                * (PI * y_component as f32 * y as f32 / height as f32).cos();
            let base = 3 * x as usize + y as usize * bytes_per_row;
            let r = srgb_to_linear(rgb[base]);
            let g = srgb_to_linear(rgb[base + 1]);
            let b = srgb_to_linear(rgb[base + 2]);
            result.0 += basis * r;
            result.1 += basis * g;
            result.2 += basis * b;
        }
    }
    let scale = normalisation / (width * height) as f32;
    (result.0 * scale, result.1 * scale, result.2 * scale)
}

/// Compute blurhash for pixels
pub fn blurhash_for_pixels(
    x_components: i32,
    y_components: i32,
    width: i32,
    height: i32,
    rgb: &[u8],
    bytes_per_row: usize,
) -> Option<String> {
    if x_components < 1 || x_components > 9 {
        return None;
    }
    if y_components < 1 || y_components > 9 {
        return None;
    }
    let num_factors = (x_components * y_components) as usize;
    let mut factors: Vec<[f32; 3]> = vec![[0.0; 3]; num_factors];

    for y in 0..y_components {
        for x in 0..x_components {
            let (r, g, b) = multiply_basis_function(x, y, width, height, rgb, bytes_per_row);
            let idx = (y * x_components + x) as usize;
            factors[idx][0] = r;
            factors[idx][1] = g;
            factors[idx][2] = b;
        }
    }

    let dc = factors[0];
    let ac = &factors[1..];
    let ac_count = num_factors - 1;

    // size flag
    let size_flag = (x_components - 1) + (y_components - 1) * 9;
    let mut ret = encode_int(size_flag, 1);

    // compute maximum value for AC components
    let mut maximum_value: f32 = 0.0;
    if ac_count > 0 {
        for acv in ac.iter() {
            for &c in acv.iter() {
                maximum_value = maximum_value.max(c.abs());
            }
        }
    }

    let quant_max_value = if ac_count > 0 {
        ((maximum_value * 166.0 - 0.5).floor().clamp(0.0, 82.0)) as i32
    } else {
        0
    };
    let norm_max_value = if ac_count > 0 {
        (quant_max_value as f32 + 1.0) / 166.0
    } else {
        1.0
    };

    ret += &encode_int(quant_max_value, 1);

    // encode DC
    let dc_value = encode_dc(dc[0], dc[1], dc[2]);
    ret += &encode_int(dc_value, 4);

    // encode AC
    for acv in ac.iter() {
        let ac_value = encode_ac(acv[0], acv[1], acv[2], norm_max_value);
        ret += &encode_int(ac_value, 2);
    }
    Some(ret)
}

/// Decode a blurhash string to a Vec<u8> representing RGB pixels
fn decode_dc(value: i32) -> [f32; 3] {
    let r = ((value >> 16) & 255) as u8;
    let g = ((value >> 8) & 255) as u8;
    let b = (value & 255) as u8;
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)]
}

fn decode_ac(value: i32, maximum_value: f32) -> [f32; 3] {
    let quant_r = (value / (19 * 19)) as i32;
    let quant_g = ((value / 19) % 19) as i32;
    let quant_b = (value % 19) as i32;
    [
        sign_pow((quant_r as f32 - 9.0) / 9.0, 2.0) * maximum_value,
        sign_pow((quant_g as f32 - 9.0) / 9.0, 2.0) * maximum_value,
        sign_pow((quant_b as f32 - 9.0) / 9.0, 2.0) * maximum_value,
    ]
}

/// Recreate pixel data from blurhash string
fn decode_blurhash(blurhash: &str, width: usize, height: usize, punch: f32) -> Option<Vec<u8>> {
    let blurhash = blurhash.trim();
    if blurhash.len() < 6 {
        return None;
    }

    // Decode size_flag
    let size_flag = decode_to_int(blurhash, 0, 1)? as usize;
    let num_y = (size_flag / 9) + 1;
    let num_x = (size_flag % 9) + 1;

    // Verify length
    let expected_length = 4 + 2 * num_x * num_y;
    if blurhash.len() != expected_length {
        return None;
    }

    // Quant max value
    let quant_max_value = decode_to_int(blurhash, 1, 2)? as f32;
    let max_ac = if quant_max_value > 0.0 {
        (quant_max_value + 1.0) / 166.0
    } else {
        1.0
    };

    // Decode DC
    let dc_value = decode_to_int(blurhash, 2, 6)?;
    let mut factors: Vec<[f32; 3]> = Vec::with_capacity(num_x * num_y);
    factors.push(decode_dc(dc_value));

    // Decode AC
    for i in 0..(num_x * num_y - 1) {
        let ac_value = decode_to_int(blurhash, 6 + 2 * i, 6 + 2 * i + 2)?;
        factors.push(decode_ac(ac_value, max_ac * punch));
    }

    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            for j in 0..num_y {
                for i in 0..num_x {
                    let basis = (PI * x as f32 * i as f32 / width as f32).cos()
                        * (PI * y as f32 * j as f32 / height as f32).cos();
                    let f = &factors[j * num_x + i];
                    r += f[0] * basis;
                    g += f[1] * basis;
                    b += f[2] * basis;
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

/// Blurhash decode as Python function - returns bytes
#[pyfunction]
fn decode_blurhash_py(
    py: Python<'_>,
    blurhash: &str,
    width: usize,
    height: usize,
    punch: Option<f32>,
) -> PyResult<Py<PyBytes>> {
    let punch = punch.unwrap_or(1.0);
    match decode_blurhash(blurhash, width, height, punch) {
        Some(bytes) => Ok(PyBytes::new(py, &bytes).into()),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to decode blurhash",
        )),
    }
}

/// Expose blurhash_for_pixels as a Python function.
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
/// Ported from:
/// bool isValidBlurhash(const char * blurhash) { ... }
pub fn is_valid_blurhash(blurhash: &str) -> bool {
    if blurhash.is_empty() || blurhash.len() < 6 {
        return false;
    }

    let size_flag = match decode_to_int(blurhash, 0, 1) {
        Some(val) => val as usize,
        None => return false,
    };

    let num_y = (size_flag / 9) + 1;
    let num_x = (size_flag % 9) + 1;

    if blurhash.len() != 4 + 2 * num_x * num_y {
        return false;
    }

    true
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
        assert_eq!(false, is_valid_blurhash(""));
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
