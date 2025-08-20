# Granite
An implementation of [Haskell's granite terminal plotting library](https://github.com/mchav/granite) in Rust.

# Supported graph types

* Scatter plots
* Histograms
* (Stacked) bar charts
* Pie charts
* Box plots
* Line charts
* Heat maps


## Examples

### Scatter plot
![Scatter Plot](https://github.com/mchav/granite-rs/blob/main/static/rust_scatter.png)

```rust
use granite_rs::*;

fn main() {
    let pts_a_x: Vec<f64> = (0..600).map(|i| i as f64 / 599.0).collect();
    let pts_a_y: Vec<f64> = (0..600).map(|i| i as f64 / 599.0).collect();
    let pts_b_x: Vec<f64> = (0..600).rev().map(|i| i as f64 / 599.0).collect();
    let pts_b_y: Vec<f64> = (0..600).map(|i| i as f64 / 599.0).collect();

    let pts_a: Vec<(f64, f64)> = pts_a_x.into_iter().zip(pts_a_y).collect();
    let pts_b: Vec<(f64, f64)> = pts_b_x.into_iter().zip(pts_b_y).collect();

    let plot_scatter = Plot { width_chars: 68, height_chars: 22, ..def_plot() };
    println!(
        "{}",
        scatter(
            "Random points",
            &[series("A", &pts_a), series("B", &pts_b)],
            &plot_scatter
        )
    );
}
```

### Bar chart
![Bar chart](https://github.com/mchav/granite/blob/main/static/bar_chart.png)

```rust
use granite_rs::*;

fn main() {
    println!(
        "{}",
        bars(
            "Sales",
            &[
                ("Q1".to_string(), 12.0),
                ("Q2".to_string(), 18.0),
                ("Q3".to_string(), 9.0),
                ("Q4".to_string(), 15.0)
            ],
            &def_plot()
        )
    );
}
```

### Stacked bar chart
![Stacked bar chart](https://github.com/mchav/granite/blob/main/static/stacked_bar.png)

### Pie chart
![Pie chart](https://github.com/mchav/granite/blob/main/static/pie_chart.png)

### Box plot
![Box plot](https://github.com/mchav/granite/blob/main/static/box_plot.png)

### Line graph
![Line graph](https://github.com/mchav/granite/blob/main/static/line_graph.png)

### Heatmap
![Heatmap](https://github.com/mchav/granite/blob/main/static/heatmap.png)
