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

    let plot_pie = Plot {
        width_chars: 46,
        height_chars: 18,
        legend_pos: LegendPos::LegendRight,
        ..def_plot()
    };
    println!(
        "{}",
        pie(
            "Share",
            &[
                ("Alpha".to_string(), 0.35),
                ("Beta".to_string(), 0.25),
                ("Gamma".to_string(), 0.20),
                ("Delta".to_string(), 0.20)
            ],
            &plot_pie
        )
    );

    let monthly = vec![
        (
            "Product A".to_string(),
            vec![(1.0, 100.0), (2.0, 120.0), (3.0, 115.0), (4.0, 140.0), (5.0, 155.0), (6.0, 148.0)],
        ),
        (
            "Product B".to_string(),
            vec![(1.0, 80.0), (2.0, 85.0), (3.0, 95.0), (4.0, 92.0), (5.0, 110.0), (6.0, 125.0)],
        ),
        (
            "Product C".to_string(),
            vec![(1.0, 60.0), (2.0, 62.0), (3.0, 70.0), (4.0, 85.0), (5.0, 82.0), (6.0, 90.0)],
        ),
    ];
    println!("{}", line_graph("Monthly Sales Trends", &monthly, &def_plot()));

    let box_data = vec![
        ("Class A".to_string(), vec![78.0, 82.0, 85.0, 88.0, 90.0, 92.0, 85.0, 87.0, 89.0, 91.0, 76.0, 94.0, 88.0]),
        ("Class B".to_string(), vec![70.0, 75.0, 72.0, 80.0, 85.0, 78.0, 82.0, 77.0, 79.0, 81.0, 74.0, 83.0]),
        ("Class C".to_string(), vec![88.0, 92.0, 95.0, 90.0, 93.0, 89.0, 91.0, 94.0, 96.0, 87.0, 90.0, 92.0]),
        ("Class D".to_string(), vec![65.0, 70.0, 72.0, 68.0, 75.0, 80.0, 73.0, 71.0, 69.0, 74.0, 77.0, 76.0]),
    ];
    println!(
        "{}",
        box_plot("Test Score Distribution by Class", &box_data, &def_plot())
    );

    let stacked = vec![
        ("Q1".to_string(), vec![("Hardware".to_string(), 120.0), ("Software".to_string(), 200.0), ("Services".to_string(), 80.0)]),
        ("Q2".to_string(), vec![("Hardware".to_string(), 135.0), ("Software".to_string(), 220.0), ("Services".to_string(), 95.0)]),
        ("Q3".to_string(), vec![("Hardware".to_string(), 110.0), ("Software".to_string(), 240.0), ("Services".to_string(), 110.0)]),
        ("Q4".to_string(), vec![("Hardware".to_string(), 145.0), ("Software".to_string(), 260.0), ("Services".to_string(), 125.0)]),
    ];
    println!(
        "{}",
        stacked_bars("Quarterly Revenue Breakdown", &stacked, &def_plot())
    );

    let matrix = vec![
        vec![ 1.0,  0.8,  0.3, -0.2,  0.1 ],
        vec![ 0.8,  1.0,  0.5, -0.1,  0.2 ],
        vec![ 0.3,  0.5,  1.0,  0.6,  0.4 ],
        vec![ -0.2, -0.1,  0.6,  1.0,  0.7 ],
        vec![ 0.1,  0.2,  0.4,  0.7,  1.0 ],
    ];
    println!("{}", heatmap("Correlation Matrix", &matrix, &def_plot()));
}
