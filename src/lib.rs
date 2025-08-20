#![allow(clippy::needless_return)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::comparison_chain)]

use std::cmp::{max, min};
use std::f64::consts::PI;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plot {
    pub width_chars: i32,
    pub height_chars: i32,
    pub left_margin: i32,
    pub bottom_margin: i32,
    pub title_margin: i32,
    pub legend_pos: LegendPos,
}

pub fn def_plot() -> Plot {
    Plot {
        width_chars: 60,
        height_chars: 20,
        left_margin: 6,
        bottom_margin: 2,
        title_margin: 1,
        legend_pos: LegendPos::LegendRight,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegendPos {
    LegendRight,
    LegendBottom
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite
}

pub fn ansi_code(color: Color) -> u8 {
    match color {
        Color::Black         => 30,
        Color::Red           => 31,
        Color::Green         => 32,
        Color::Yellow        => 33,
        Color::Blue          => 34,
        Color::Magenta       => 35,
        Color::Cyan          => 36,
        Color::White         => 37,
        Color::BrightBlack   => 90,
        Color::BrightRed     => 91,
        Color::BrightGreen   => 92,
        Color::BrightYellow  => 93,
        Color::BrightBlue    => 94,
        Color::BrightMagenta => 95,
        Color::BrightCyan    => 96,
        Color::BrightWhite   => 97,
        _                    => 39,
    }
}

fn paint(color : Color, ch : char) -> String {
    if ch == ' ' {
        " ".to_string()
    } else {
        format!("\x1b[{}m{}\x1b[0m", ansi_code(color), ch)
    }
}

fn palette_colors() -> Vec<Color> {
    vec![
        Color::BrightBlue,
        Color::BrightMagenta,
        Color::BrightCyan,
        Color::BrightGreen,
        Color::BrightYellow,
        Color::BrightRed,
        Color::BrightWhite,
        Color::BrightBlack,
    ]
}

fn pie_colors() -> Vec<Color> {
    vec![
        Color::BrightRed,
        Color::BrightGreen,
        Color::BrightYellow,
        Color::BrightBlue,
        Color::BrightMagenta,
        Color::BrightCyan,
        Color::BrightWhite,
        Color::BrightBlack,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pat {
    Solid,
    Checker,
    DiagA,
    DiagB,
    Sparse,
}

fn ink(p: Pat, x: i32, y: i32) -> bool {
    match p {
        Pat::Solid => true,
        Pat::Checker => ((x ^ y) & 1) == 0,
        Pat::DiagA => ((x + y) % 3) != 1,
        Pat::DiagB => ((x - y) % 3) != 1,
        Pat::Sparse => (x & 1 == 0) && (y % 3 == 0),
    }
}

fn palette() -> Vec<Pat> {
    vec![Pat::Solid, Pat::Checker, Pat::DiagA, Pat::DiagB, Pat::Sparse]
}

#[derive(Clone)]
struct Array2D<T: Clone> {
    w: i32,
    h: i32,
    data: Vec<Vec<T>>,
}

fn new_a2d<T: Clone>(w: i32, h: i32, v: T) -> Array2D<T> {
    Array2D {
        w,
        h,
        data: vec![vec![v; w as usize]; h as usize],
    }
}

fn get_a2d<T: Clone>(a: &Array2D<T>, x: i32, y: i32) -> T {
    a.data[y as usize][x as usize].clone()
}

fn set_a2d<T: Clone>(a: &mut Array2D<T>, x: i32, y: i32, v: T) {
    a.data[y as usize][x as usize] = v;
}

fn to_bit(ry: i32, rx: i32) -> u16 {
    match (ry, rx) {
        (0, 0) => 1,
        (1, 0) => 2,
        (2, 0) => 4,
        (3, 0) => 64,
        (0, 1) => 8,
        (1, 1) => 16,
        (2, 1) => 32,
        (3, 1) => 128,
        _ => 0,
    }
}

#[derive(Clone)]
struct Canvas {
    cw: i32,
    ch: i32,
    buffer: Array2D<u16>,
    cbuf: Array2D<Option<Color>>,
}

fn new_canvas(w: i32, h: i32) -> Canvas {
    Canvas {
        cw: w,
        ch: h,
        buffer: new_a2d(w, h, 0u16),
        cbuf: new_a2d(w, h, None),
    }
}

fn set_dot_c(mut c: Canvas, xdot: i32, ydot: i32, mcol: Option<Color>) -> Canvas {
    if xdot < 0 || ydot < 0 || xdot >= c.cw * 2 || ydot >= c.ch * 4 {
        return c;
    }
    let cx = xdot / 2;
    let cy = ydot / 4;
    let rx = xdot - 2 * cx;
    let ry = ydot - 4 * cy;
    let b = to_bit(ry, rx);
    let m = get_a2d(&c.buffer, cx, cy);
    set_a2d(&mut c.buffer, cx, cy, m | b);
    if let Some(col) = mcol {
        set_a2d(&mut c.cbuf, cx, cy, Some(col));
    }
    c
}

fn fill_dots_c(
    (x0, y0): (i32, i32),
    (x1, y1): (i32, i32),
    p: &dyn Fn(i32, i32) -> bool,
    mcol: Option<Color>,
    c0: Canvas,
) -> Canvas {
    let xs = (max(0, x0))..=min(c0.cw * 2 - 1, x1);
    let ys = (max(0, y0))..=min(c0.ch * 4 - 1, y1);
    let mut c = c0;
    for y in ys {
        for x in xs.clone() {
            if p(x, y) {
                c = set_dot_c(c, x, y, mcol);
            }
        }
    }
    c
}

fn render_canvas(c: &Canvas) -> String {
    fn glyph(m: u16) -> char {
        if m == 0 {
            ' '
        } else {
            char::from_u32(0x2800 + m as u32).unwrap_or(' ')
        }
    }
    let mut out = String::new();
    for y in 0..c.ch {
        for x in 0..c.cw {
            let m = get_a2d(&c.buffer, x, y);
            let ch = glyph(m);
            let mc = get_a2d(&c.cbuf, x, y);
            if let Some(col) = mc {
                out.push_str(&paint(col, ch));
            } else {
                out.push(ch);
            }
        }
        out.push('\n');
    }
    out
}

fn wcswidth(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut acc = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'm' {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'm' {
                i += 1;
            }
        } else {
            let ch = s[i..].chars().next().unwrap();
            acc += 1;
            i += ch.len_utf8();
        }
    }
    acc
}

fn justify_right(n: i32, s: &str) -> String {
    let w = wcswidth(s) as i32;
    if n > w {
        format!("{}{}", " ".repeat((n - w) as usize), s)
    } else {
        s.to_string()
    }
}

fn fmt(v: f64) -> String {
    let av = v.abs();
    if av >= 1000.0 || (av < 0.01 && v != 0.0) {
        format!("{:.1e}", v)
    } else {
        format!("{:.1}", v)
    }
}

fn draw_frame(_cfg: &Plot, title_str: &str, content_with_axes: &str, legend_block: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if !title_str.is_empty() {
        parts.push(title_str);
    }
    if !content_with_axes.is_empty() {
        parts.push(content_with_axes);
    }
    if !legend_block.is_empty() {
        parts.push(legend_block);
    }
    parts.join("\n")
}

fn place_labels(mut base: String, off: i32, xs: &[(i32, String)]) -> String {
    let mut chars: Vec<char> = base.chars().collect();
    for (x, s) in xs {
        let i = (off + *x).max(0) as usize;
        let sw = wcswidth(s);
        if i + sw > chars.len() {
            chars.resize(i + sw, ' ');
        }
        for (j, ch) in s.chars().enumerate() {
            if i + j < chars.len() {
                chars[i + j] = ch;
            }
        }
    }
    base.clear();
    for ch in chars {
        base.push(ch);
    }
    base
}

fn axisify(cfg: &Plot, c: &Canvas, (xmin, xmax): (f64, f64), (ymin, ymax): (f64, f64)) -> String {
    let plot_w = c.cw;
    let plot_h = c.ch;
    let left = cfg.left_margin;
    let pad = " ".repeat(left as usize);

    let y_ticks = vec![(0, ymax), (plot_h / 2, (ymin + ymax) / 2.0), (plot_h - 1, ymin)];
    let mut y_labels: Vec<String> = vec![pad.clone(); plot_h as usize];
    for (row, v) in y_ticks {
        if row >= 0 && row < plot_h {
            y_labels[row as usize] = justify_right(left, &fmt(v));
        }
    }

    let rendered = render_canvas(c);
    let canvas_lines: Vec<&str> = rendered.lines().collect();
    let mut attach_y: Vec<String> = Vec::with_capacity(canvas_lines.len());
    for (lbl, line) in y_labels.iter().zip(canvas_lines.iter()) {
        attach_y.push(format!("{}│{}", lbl, line));
    }

    let x_bar = format!("{}│{}", pad, "─".repeat(plot_w as usize));
    let x_lbls = vec![(0, xmin), (plot_w / 2, (xmin + xmax) / 2.0), (plot_w - 1, xmax)];
    let mut x_line = " ".repeat((left + 1 + plot_w) as usize);
    let x_pairs: Vec<(i32, String)> = x_lbls
        .into_iter()
        .map(|(x, v)| (x, fmt(v)))
        .collect();
    x_line = place_labels(x_line, left + 1, &x_pairs);

    let mut out = String::new();
    for row in &attach_y {
        out.push_str(row);
        out.push('\n');
    }
    out.push_str(&x_bar);
    out.push('\n');
    out.push_str(&x_line);
    out
}

fn render_row(cells: &[(char, Option<Color>)]) -> String {
    let mut s = String::new();
    for (ch, mc) in cells {
        if let Some(col) = mc {
            s.push_str(&paint(*col, *ch));
        } else {
            s.push(*ch);
        }
    }
    s
}

fn axisify_grid(
    cfg: &Plot,
    grid: &[Vec<(char, Option<Color>)>],
    (xmin, xmax): (f64, f64),
    (ymin, ymax): (f64, f64),
) -> String {
    let plot_h = grid.len() as i32;
    let plot_w = if plot_h == 0 { 0 } else { grid[0].len() as i32 };
    let left = cfg.left_margin;
    let pad = " ".repeat(left as usize);

    let y_ticks = vec![(0, ymax), (plot_h / 2, (ymin + ymax) / 2.0), (plot_h - 1, ymin)];
    let mut y_labels: Vec<String> = vec![pad.clone(); plot_h as usize];
    for (row, v) in y_ticks {
        if row >= 0 && row < plot_h {
            y_labels[row as usize] = justify_right(left, &fmt(v));
        }
    }

    let mut attach_y: Vec<String> = Vec::with_capacity(grid.len());
    for (lbl, row) in y_labels.iter().zip(grid.iter()) {
        attach_y.push(format!("{}│{}", lbl, render_row(row)));
    }

    let x_bar = format!("{}│{}", pad, "─".repeat(plot_w as usize));
    let x_lbls = vec![(0, xmin), (plot_w / 2, (xmin + xmax) / 2.0), (plot_w - 1, xmax)];
    let mut x_line = " ".repeat((left + 1 + plot_w) as usize);
    let x_pairs: Vec<(i32, String)> = x_lbls
        .into_iter()
        .map(|(x, v)| (x, fmt(v)))
        .collect();
    x_line = place_labels(x_line, left + 1, &x_pairs);

    let mut out = String::new();
    for row in &attach_y {
        out.push_str(row);
        out.push('\n');
    }
    out.push_str(&x_bar);
    out.push('\n');
    out.push_str(&x_line);
    out
}

fn legend_block(pos: LegendPos, width: i32, entries: &[(String, Pat, Color)]) -> String {
    match pos {
        LegendPos::LegendBottom => {
            let cells: Vec<String> = entries
                .iter()
                .map(|(name, pat, col)| format!("{} {}", sample(*pat, *col), name))
                .collect();
            let line = cells.join("   ");
            let vis = wcswidth(&line) as i32;
            let pad = if vis < width {
                " ".repeat(((width - vis) / 2) as usize)
            } else {
                String::new()
            };
            format!("{}{}", pad, line)
        }
        LegendPos::LegendRight => {
            let mut s = String::new();
            for (name, pat, col) in entries {
                let _ = writeln!(&mut s, "{} {}", sample(*pat, *col), name);
            }
            s
        }
    }
}

fn sample(p: Pat, col: Color) -> String {
    let mut c = new_canvas(1, 1);
    for y in 0..4 {
        for x in 0..2 {
            if ink(p, x, y) {
                c = set_dot_c(c, x % 2, y % 4, Some(col));
            }
        }
    }
    let mut s = render_canvas(&c);
    while s.ends_with('\n') {
        s.pop();
    }
    s
}

fn clamp<T: PartialOrd>(low: T, high: T, x: T) -> T {
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

const EPS: f64 = 1e-12;

fn bounds_xy(pts: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let xs = pts.iter().map(|(x, _)| *x);
    let ys = pts.iter().map(|(_, y)| *y);
    let xmin = xs.clone().fold(f64::INFINITY, f64::min);
    let xmax = xs.fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.clone().fold(f64::INFINITY, f64::min);
    let ymax = ys.fold(f64::NEG_INFINITY, f64::max);
    let padx = (xmax - xmin) * 0.05 + 1e-9;
    let pady = (ymax - ymin) * 0.05 + 1e-9;
    (xmin - padx, xmax + padx, ymin - pady, ymax + pady)
}

fn mod_f(a: f64, m: f64) -> f64 {
    a - (a / m).floor() * m
}

pub fn series(name: &str, pts: &[(f64, f64)]) -> (String, Vec<(f64, f64)>) {
    (name.to_string(), pts.to_vec())
}

pub fn scatter(title: &str, sers: &[(String, Vec<(f64, f64)>)], cfg: &Plot) -> String {
    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;
    let mut plot_c = new_canvas(w_c, h_c);

    let all: Vec<(f64, f64)> = sers.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let (xmin, xmax, ymin, ymax) = if all.is_empty() {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        bounds_xy(&all)
    };
    let sx = |x: f64| -> i32 {
        clamp(
            0,
            w_c * 2 - 1,
            (((x - xmin) / (xmax - xmin + EPS) * (w_c * 2 - 1) as f64).round()) as i32,
        )
    };
    let sy = |y: f64| -> i32 {
        clamp(
            0,
            h_c * 4 - 1,
            (((ymax - y) / (ymax - ymin + EPS) * (h_c * 4 - 1) as f64).round()) as i32,
        )
    };

    let mut pats = palette().into_iter().cycle();
    let mut cols = palette_colors().into_iter().cycle();
    let mut with_sty: Vec<(&str, &[(f64, f64)], Pat, Color)> = Vec::new();
    for (n, pts) in sers {
        with_sty.push((
            n.as_str(),
            pts.as_slice(),
            pats.next().unwrap(),
            cols.next().unwrap(),
        ));
    }

    for (_name, pts, pat, col) in &with_sty {
        for (x, y) in *pts {
            let xd = sx(*x);
            let yd = sy(*y);
            if ink(*pat, xd, yd) {
                plot_c = set_dot_c(plot_c, xd, yd, Some(*col));
            }
        }
    }

    let ax = axisify(cfg, &plot_c, (xmin, xmax), (ymin, ymax));
    let legend = legend_block(
        cfg.legend_pos,
        cfg.left_margin + cfg.width_chars,
        &with_sty
            .iter()
            .map(|(n, _, p, c)| (n.to_string(), *p, *c))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

fn block_char(n: i32) -> char {
    match clamp(0, 8, n) {
        0 => ' ',
        1 => '▁',
        2 => '▂',
        3 => '▃',
        4 => '▄',
        5 => '▅',
        6 => '▆',
        7 => '▇',
        _ => '█',
    }
}

fn col_glyphs(hc: i32, frac: f64) -> String {
    let total = hc * 8;
    let ticks = clamp(0, total, (frac * (total as f64)).round() as i32);
    let full = ticks / 8;
    let rem8 = ticks - full * 8;
    let top_pad = hc - full - if rem8 > 0 { 1 } else { 0 };
    let mut s = String::new();
    s.push_str(&" ".repeat(top_pad as usize));
    if rem8 > 0 {
        s.push(block_char(rem8));
    }
    s.push_str(&"█".repeat(full as usize));
    s
}

fn resample_to_width(w: i32, xs: &[f64]) -> Vec<f64> {
    if w <= 0 {
        return vec![];
    }
    if xs.is_empty() {
        return vec![0.0; w as usize];
    }
    let n = xs.len() as i32;
    if n == w {
        return xs.to_vec();
    } else if n > w {
        let g = ((n as f64) / (w as f64)).ceil() as i32;
        let mut out = Vec::with_capacity(w as usize);
        for i in 0..w {
            let start = (i * g) as usize;
            let end = min((start + g as usize) as i32, n) as usize;
            let slice = &xs[start..end];
            let v = if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f64>() / (slice.len() as f64)
            };
            out.push(v);
        }
        return out;
    } else {
        let base = w / n;
        let extra = w - base * n;
        let mut out = Vec::with_capacity(w as usize);
        for (i, v) in xs.iter().enumerate() {
            let i = i as i32;
            let count = base + if i < extra { 1 } else { 0 };
            for _ in 0..count {
                out.push(*v);
            }
        }
        return out;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bins {
    pub n_bins: i32,
    pub lo: f64,
    pub hi: f64,
}

pub fn bins(n: i32, a: f64, b: f64) -> Bins {
    Bins {
        n_bins: max(1, n),
        lo: a.min(b),
        hi: a.max(b),
    }
}

pub fn histogram(title: &str, b: Bins, xs: &[f64], cfg: &Plot) -> String {
    let n = b.n_bins;
    let a = b.lo;
    let bb = b.hi;
    let step = (bb - a) / (n as f64);
    let bin_ix = |x: f64| -> i32 { clamp(0, n - 1, ((x - a) / step).floor() as i32) };
    let mut counts = vec![0i32; n as usize];
    for &x in xs {
        if x >= a && x <= bb {
            let i = bin_ix(x) as usize;
            counts[i] += 1;
        }
    }
    let max_c = counts.iter().cloned().fold(1, i32::max) as f64;
    let fracs0: Vec<f64> = counts.iter().map(|&c| (c as f64) / max_c).collect();

    let w_data = cfg.width_chars;
    let h_c = cfg.height_chars;
    let cols_f = resample_to_width(w_data, &fracs0);

    let data_cols: Vec<(String, Option<Color>)> = cols_f
        .iter()
        .map(|&f| (col_glyphs(h_c, f), Some(Color::BrightCyan)))
        .collect();
    let gutter_col = (String::from(" ".repeat(h_c as usize)), None::<Color>);

    let mut columns: Vec<(String, Option<Color>)> = Vec::new();
    for (i, col) in data_cols.iter().enumerate() {
        if i > 0 {
            columns.push(gutter_col.clone());
        }
        columns.push(col.clone());
    }

    let mut grid: Vec<Vec<(char, Option<Color>)>> = Vec::new();
    for y in 0..h_c {
        let mut row: Vec<(char, Option<Color>)> = Vec::new();
        for (glyphs, mc) in &columns {
            let ch = glyphs.chars().nth(y as usize).unwrap_or(' ');
            row.push((ch, *mc));
        }
        grid.push(row);
    }

    let ax = axisify_grid(cfg, &grid, (a, bb), (0.0, counts.iter().cloned().fold(1, i32::max) as f64));
    let legend_width = cfg.left_margin + 1 + if grid.is_empty() { 0 } else { grid[0].len() as i32 };
    let legend = legend_block(cfg.legend_pos, legend_width, &[("count".to_string(), Pat::Solid, Color::BrightCyan)]);
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

pub fn bars(title: &str, kvs: &[(String, f64)], cfg: &Plot) -> String {
    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;
    let vals: Vec<f64> = kvs.iter().map(|(_, v)| *v).collect();
    let vmax = vals
        .iter()
        .map(|v| v.abs())
        .fold(1e-12f64, |a, b| a.max(b));

    let mut cats: Vec<(String, f64, Color)> = Vec::new();
    let mut cols = palette_colors().into_iter().cycle();
    for (name, v) in kvs {
        cats.push((name.clone(), v.abs() / vmax, cols.next().unwrap()));
    }
    let n_cats = cats.len() as i32;

    let base = if n_cats == 0 { 0 } else { w_c / n_cats };
    let extra = if n_cats == 0 { 0 } else { w_c - base * n_cats };
    let widths: Vec<i32> = (0..n_cats)
        .map(|i| base + if i < extra { 1 } else { 0 })
        .collect();

    let mut cat_groups: Vec<Vec<(String, Option<Color>)>> = Vec::new();
    for (i, (_name, f, col)) in cats.iter().enumerate() {
        let w = widths[i as usize];
        cat_groups.push(vec![(col_glyphs(h_c, *f), Some(*col)); w as usize]);
    }

    let gutter_col = (String::from(" ".repeat(h_c as usize)), None::<Color>);
    let mut columns: Vec<(String, Option<Color>)> = Vec::new();
    for (i, grp) in cat_groups.iter().enumerate() {
        if i > 0 {
            columns.push(gutter_col.clone());
        }
        columns.extend(grp.clone());
    }

    let mut grid: Vec<Vec<(char, Option<Color>)>> = Vec::new();
    for y in 0..h_c {
        let mut row: Vec<(char, Option<Color>)> = Vec::new();
        for (glyphs, mc) in &columns {
            let ch = glyphs.chars().nth(y as usize).unwrap_or(' ');
            row.push((ch, *mc));
        }
        grid.push(row);
    }

    let ax = axisify_grid(
        cfg,
        &grid,
        (0.0, (max(1, n_cats)) as f64),
        (0.0, vmax),
    );
    let legend_width = cfg.left_margin + 1 + if grid.is_empty() { 0 } else { grid[0].len() as i32 };
    let legend = legend_block(
        cfg.legend_pos,
        legend_width,
        &cats.iter()
            .map(|(name, _, col)| (name.clone(), Pat::Checker, *col))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

fn normalize(parts: &[(String, f64)]) -> Vec<(String, f64)> {
    let s = parts.iter().map(|(_, v)| v.abs()).sum::<f64>() + 1e-12;
    parts
        .iter()
        .map(|(n, v)| (n.clone(), (v / s).max(0.0)))
        .collect()
}

fn angle_within(ang: f64, a0: f64, a1: f64) -> bool {
    if a1 >= a0 {
        ang >= a0 && ang <= a1
    } else {
        ang >= a0 || ang <= a1
    }
}

pub fn pie(title: &str, parts0: &[(String, f64)], cfg: &Plot) -> String {
    let parts = normalize(parts0);
    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;
    let mut plot_c = new_canvas(w_c, h_c);
    let w_dots = w_c * 2;
    let h_dots = h_c * 4;
    let r = min(w_dots / 2 - 2, h_dots / 2 - 2);
    let cx = w_dots / 2;
    let cy = h_dots / 2;
    let to_ang = |p: f64| p * 2.0 * PI;

    let mut wedges: Vec<f64> = Vec::with_capacity(parts.len() + 1);
    wedges.push(0.0);
    for (_, p) in &parts {
        let prev = *wedges.last().unwrap();
        wedges.push(prev + to_ang(*p));
    }
    let angles: Vec<(f64, f64)> = wedges.iter().copied().zip(wedges.iter().copied().skip(1)).collect();

    let names: Vec<String> = parts.iter().map(|(n, _)| n.clone()).collect();
    let mut cols = pie_colors().into_iter().cycle();
    let with_p: Vec<(String, (f64, f64), Color)> = names
        .into_iter()
        .zip(angles.into_iter())
        .map(|(n, ang)| (n, ang, cols.next().unwrap()))
        .collect();

    for (_name, (a0, a1), col) in &with_p {
        let inside = |x: i32, y: i32| {
            let dx = (x - cx) as f64;
            let dy = (cy - y) as f64;
            let rr2 = dx * dx + dy * dy;
            let r2 = (r * r) as f64;
            if rr2 > r2 {
                return false;
            }
            let ang = mod_f(dy.atan2(dx), 2.0 * PI);
            angle_within(ang, *a0, *a1)
        };
        plot_c = fill_dots_c((cx - r, cy - r), (cx + r, cy + r), &inside, Some(*col), plot_c);
    }

    let ax = axisify(cfg, &plot_c, (0.0, 1.0), (0.0, 1.0));
    let legend = legend_block(
        cfg.legend_pos,
        cfg.left_margin + cfg.width_chars,
        &with_p
            .iter()
            .map(|(n, _, col)| (n.clone(), Pat::Solid, *col))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

fn line_dots_c((x0, y0): (i32, i32), (x1, y1): (i32, i32), mcol: Option<Color>, c0: Canvas) -> Canvas {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut c = c0;

    loop {
        c = set_dot_c(c, x, y, mcol);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    c
}

pub fn line_graph(title: &str, sers: &[(String, Vec<(f64, f64)>)], cfg: &Plot) -> String {
    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;
    let mut plot_c = new_canvas(w_c, h_c);

    let all: Vec<(f64, f64)> = sers.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let (xmin, xmax, ymin, ymax) = if all.is_empty() {
        (0.0, 1.0, 0.0, 1.0)
    } else {
        bounds_xy(&all)
    };
    let sx = |x: f64| -> i32 {
        clamp(
            0,
            w_c * 2 - 1,
            (((x - xmin) / (xmax - xmin + EPS) * (w_c * 2 - 1) as f64).round()) as i32,
        )
    };
    let sy = |y: f64| -> i32 {
        clamp(
            0,
            h_c * 4 - 1,
            (((ymax - y) / (ymax - ymin + EPS) * (h_c * 4 - 1) as f64).round()) as i32,
        )
    };

    let mut cols = palette_colors().into_iter().cycle();
    let with_sty: Vec<(&String, &Vec<(f64, f64)>, Color)> = sers
        .iter()
        .map(|(n, pts)| (n, pts, cols.next().unwrap()))
        .collect();

    for (_name, pts, col) in &with_sty {
        if pts.len() < 2 {
            continue;
        }
        let mut sorted = pts.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in sorted.windows(2) {
            let (x1, y1) = w[0];
            let (x2, y2) = w[1];
            plot_c = line_dots_c((sx(x1), sy(y1)), (sx(x2), sy(y2)), Some(*col), plot_c);
        }
    }

    let ax = axisify(cfg, &plot_c, (xmin, xmax), (ymin, ymax));
    let legend = legend_block(
        cfg.legend_pos,
        cfg.left_margin + cfg.width_chars,
        &with_sty
            .iter()
            .map(|(n, _pts, col)| (n.to_string(), Pat::Solid, *col))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

fn quartiles(xs: &[f64]) -> (f64, f64, f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n < 5 {
        let m = s.iter().sum::<f64>() / (n as f64);
        return (m, m, m, m, m);
    }
    let q1_idx = n / 4;
    let q2_idx = n / 2;
    let q3_idx = (3 * n) / 4;
    let get_idx = |i: usize| if i < n { s[i] } else { s[n - 1] };
    (s[0], get_idx(q1_idx), get_idx(q2_idx), get_idx(q3_idx), s[n - 1])
}

pub fn box_plot(title: &str, datasets: &[(String, Vec<f64>)], cfg: &Plot) -> String {
    let w_c = cfg.width_chars as usize;
    let h_c = cfg.height_chars as usize;

    let stats: Vec<(String, (f64, f64, f64, f64, f64))> =
        datasets.iter().map(|(n, v)| (n.clone(), quartiles(v))).collect();

    let all_vals: Vec<f64> = datasets.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let ymin = if all_vals.is_empty() {
        0.0
    } else {
        let m = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        m - m.abs() * 0.1
    };
    let ymax = if all_vals.is_empty() {
        1.0
    } else {
        let m = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + m.abs() * 0.1
    };

    let n_boxes = datasets.len().max(1);
    let box_width = max(1, (cfg.width_chars as usize) / (n_boxes * 2));
    let spacing = if n_boxes <= 1 {
        0
    } else {
        (cfg.width_chars as usize - box_width * n_boxes) / (n_boxes - 1)
    };

    let scale_y = |v: f64| -> i32 {
        clamp(
            0,
            cfg.height_chars - 1,
            (((ymax - v) / (ymax - ymin + EPS) * (cfg.height_chars - 1) as f64).round()) as i32,
        )
    };

    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w_c]; h_c];

    let draw_v_line = |g: &mut Vec<Vec<(char, Option<Color>)>>, x: usize, y1: i32, y2: i32, ch: char, col: Color| {
        let ys = min(y1, y2)..=max(y1, y2);
        for y in ys {
            let yi = y as usize;
            if yi < g.len() && x < g[0].len() {
                g[yi][x] = (ch, Some(col));
            }
        }
    };
    let draw_h_line = |g: &mut Vec<Vec<(char, Option<Color>)>>, x1: usize, x2: usize, y: i32, ch: char, col: Color| {
        let xs = min(x1, x2)..=max(x1, x2);
        let yi = y as usize;
        if yi < g.len() {
            for x in xs {
                if x < g[0].len() {
                    g[yi][x] = (ch, Some(col));
                }
            }
        }
    };
    let set_grid_char = |g: &mut Vec<Vec<(char, Option<Color>)>>, x: usize, y: i32, ch: char, col: Color| {
        let yi = y as usize;
        if yi < g.len() && x < g[0].len() {
            g[yi][x] = (ch, Some(col));
        }
    };

    for (idx, (_name, (minv, q1, med, q3, maxv))) in stats.iter().enumerate() {
        let x_start = idx * (box_width + spacing);
        let x_mid = x_start + box_width / 2;
        let x_end = x_start + box_width.saturating_sub(1);
        let min_row = scale_y(*minv);
        let q1_row = scale_y(*q1);
        let med_row = scale_y(*med);
        let q3_row = scale_y(*q3);
        let max_row = scale_y(*maxv);
        let col = pie_colors()[idx % pie_colors().len()];

        draw_v_line(&mut grid, x_mid, min_row, q1_row, '│', col);
        draw_v_line(&mut grid, x_mid, q3_row, max_row, '│', col);
        draw_h_line(&mut grid, x_start, x_end, q1_row, '─', col);
        draw_h_line(&mut grid, x_start, x_end, q3_row, '─', col);
        draw_v_line(&mut grid, x_start, q1_row, q3_row, '│', col);
        draw_v_line(&mut grid, x_end, q1_row, q3_row, '│', col);
        draw_h_line(&mut grid, x_start, x_end, med_row, '═', col);
        set_grid_char(&mut grid, x_mid, min_row, '┬', col);
        set_grid_char(&mut grid, x_mid, max_row, '┴', col);
    }

    let ax = axisify_grid(
        cfg,
        &grid,
        (0.0, datasets.len().max(1) as f64),
        (ymin, ymax),
    );
    let legend = legend_block(
        cfg.legend_pos,
        cfg.left_margin + cfg.width_chars,
        &stats
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), Pat::Solid, pie_colors()[i % pie_colors().len()]))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}

pub fn heatmap(title: &str, matrix: &[Vec<f64>], cfg: &Plot) -> String {
    let rows = matrix.len() as i32;
    let cols = if rows == 0 { 0 } else { matrix[0].len() as i32 };

    let all_vals: Vec<f64> = matrix.iter().flat_map(|r| r.iter()).copied().collect();
    let vmin = if all_vals.is_empty() {
        0.0
    } else {
        all_vals.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let vmax = if all_vals.is_empty() {
        1.0
    } else {
        all_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let vrange = vmax - vmin + EPS;

    let intensity_colors = vec![
        Color::Blue,
        Color::Cyan,
        Color::BrightCyan,
        Color::Green,
        Color::BrightGreen,
        Color::Yellow,
        Color::BrightYellow,
        Color::Red,
        Color::BrightRed,
        Color::Magenta,
        Color::BrightMagenta,
    ];
    let color_for_value = |v: f64| -> Color {
        let norm = clamp(0.0, 1.0, (v - vmin) / vrange);
        let idx = clamp(
            0,
            (intensity_colors.len() - 1) as i32,
            (norm * ((intensity_colors.len() - 1) as f64)).floor() as i32,
        ) as usize;
        intensity_colors[idx]
    };

    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;

    let mut resampled: Vec<Vec<f64>> = vec![vec![0.0; w_c as usize]; h_c as usize];
    for i in 0..h_c {
        for j in 0..w_c {
            let ri = if h_c == 1 {
                0.0
            } else {
                (i as f64) * ((rows - 1).max(0) as f64) / ((h_c - 1).max(1) as f64)
            };
            let ci = if w_c == 1 {
                0.0
            } else {
                (j as f64) * ((cols - 1).max(0) as f64) / ((w_c - 1).max(1) as f64)
            };
            let r0 = clamp(0, rows - 1, ri.floor() as i32) as usize;
            let r1 = clamp(0, rows - 1, ri.ceil() as i32) as usize;
            let c0 = clamp(0, cols - 1, ci.floor() as i32) as usize;
            let c1 = clamp(0, cols - 1, ci.ceil() as i32) as usize;

            let v00 = if rows == 0 || cols == 0 { 0.0 } else { matrix[r0][c0] };
            let v01 = if rows == 0 || cols == 0 { 0.0 } else { matrix[r0][c1] };
            let v10 = if rows == 0 || cols == 0 { 0.0 } else { matrix[r1][c0] };
            let v11 = if rows == 0 || cols == 0 { 0.0 } else { matrix[r1][c1] };
            let fr = ri - (r0 as f64);
            let fc = ci - (c0 as f64);
            let v0 = v00 * (1.0 - fc) + v01 * fc;
            let v1 = v10 * (1.0 - fc) + v11 * fc;
            resampled[i as usize][j as usize] = v0 * (1.0 - fr) + v1 * fr;
        }
    }

    let mut grid: Vec<Vec<(char, Option<Color>)>> = Vec::new();
    for row in &resampled {
        let mut r: Vec<(char, Option<Color>)> = Vec::new();
        for &val in row {
            r.push(('█', Some(color_for_value(val))));
        }
        grid.push(r);
    }

    let ax = axisify_grid(cfg, &grid, (0.0, cols as f64), (rows as f64, 0.0));
    let legend_colors: Vec<Color> = intensity_colors.into_iter().take(9).collect();
    let mut gradient_legend = String::from("Min ");
    for col in legend_colors {
        gradient_legend.push_str(&paint(col, '█'));
    }
    gradient_legend.push_str(" Max");

    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &gradient_legend)
}

pub fn stacked_bars(
    title: &str,
    categories: &[(String, Vec<(String, f64)>)],
    cfg: &Plot,
) -> String {
    let w_c = cfg.width_chars;
    let h_c = cfg.height_chars;

    let series_names: Vec<String> = if categories.is_empty() || categories[0].1.is_empty() {
        vec![]
    } else {
        categories[0].1.iter().map(|(n, _)| n.clone()).collect()
    };

    let totals: Vec<f64> = categories
        .iter()
        .map(|(_, series)| series.iter().map(|(_, v)| *v).sum::<f64>())
        .collect();
    let max_height = totals
        .iter()
        .fold(1e-12f64, |a, b| a.max(*b));

    let n_cats = categories.len() as i32;
    let base = if n_cats == 0 { 0 } else { w_c / n_cats };
    let extra = if n_cats == 0 { 0 } else { w_c - base * n_cats };
    let widths: Vec<i32> = (0..n_cats)
        .map(|i| base + if i < extra { 1 } else { 0 })
        .collect();

    let mut cols = palette_colors().into_iter().cycle();
    let mut series_colors: Vec<(String, Color)> = Vec::new();
    for n in &series_names {
        series_colors.push((n.clone(), cols.next().unwrap()));
    }

    let mut columns: Vec<Vec<(char, Option<Color>)>> = Vec::new();
    let gutter_col: Vec<(char, Option<Color>)> = (0..h_c).map(|_| (' ', None)).collect();

    for ((_, series), width) in categories.iter().zip(widths.iter()) {
        let mut cum = vec![0.0f64];
        for (_, v) in series {
            cum.push(cum.last().unwrap() + (*v / max_height));
        }
        let segments: Vec<(&str, f64, f64)> = series
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.as_str(), cum[i], cum[i + 1]))
            .collect();

        let make_column = |y: i32| -> (char, Option<Color>) {
            let height_from_bottom = ((h_c - y) as f64) / (h_c as f64);
            let mut cell: (char, Option<Color>) = (' ', None);
            for (name, bottom, top) in &segments {
                if height_from_bottom > *bottom && height_from_bottom <= *top {
                    let col = series_colors
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, c)| *c);
                    cell = ('█', col);
                    break;
                }
            }
            cell
        };

        let one_bar: Vec<(char, Option<Color>)> = (0..h_c).map(|y| make_column(y)).collect();
        for _ in 0..*width {
            columns.push(one_bar.clone());
        }
        columns.push(gutter_col.clone());
    }
    if !columns.is_empty() {
        columns.pop();
    }

    let mut grid: Vec<Vec<(char, Option<Color>)>> = Vec::new();
    for y in 0..h_c {
        let mut row: Vec<(char, Option<Color>)> = Vec::new();
        for col in &columns {
            row.push(col[y as usize]);
        }
        grid.push(row);
    }

    let ax = axisify_grid(
        cfg,
        &grid,
        (0.0, (max(1, n_cats)) as f64),
        (0.0, max_height),
    );
    let legend = legend_block(
        cfg.legend_pos,
        cfg.left_margin + 1 + if grid.is_empty() { 0 } else { grid[0].len() as i32 },
        &series_colors
            .iter()
            .map(|(name, col)| (name.clone(), Pat::Solid, *col))
            .collect::<Vec<_>>(),
    );
    let titled = if title.is_empty() { "" } else { title };
    draw_frame(cfg, titled, &ax, &legend)
}
