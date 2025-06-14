use chrono::Local;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    symbols::line,
    widgets::{
        block::{Position, Title},
        *,
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    io,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
struct ModelCall {
    timestamp: String,
    call_type: String,
    category: String,
    model_name: String,
    tokens: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost: f64,
    energy_joules: f64,
    execution_time_ms: u64,
    success: bool,
}

#[derive(Clone, Debug)]
struct SocioeconomicTransaction {
    timestamp: String,
    transaction_type: String,
    description: String,
    amount: f64,
    expected_empowerment_gain: f64,
    actual_empowerment_gain: f64,
    transaction_id: String,
}

#[derive(Clone)]
struct Model {
    name: String,
    cost_per_kilo_token: f64,
    joules_per_input_token: f64,
    joules_per_output_token: f64,
    pue_factor: f64,
}

impl Model {
    fn calculate_energy(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let base_energy = (input_tokens as f64 * self.joules_per_input_token)
            + (output_tokens as f64 * self.joules_per_output_token);
        base_energy * self.pue_factor
    }

    fn max_tokens_from_funds(&self, available_funds: f64) -> u64 {
        if self.cost_per_kilo_token <= 0.0 {
            return u64::MAX;
        }
        ((available_funds / self.cost_per_kilo_token) * 1000.0) as u64
    }

    fn energy_for_max_tokens(&self, max_tokens: u64) -> f64 {
        // Assume 60% input, 40% output ratio for estimation
        let input_tokens = (max_tokens as f64 * 0.6) as u64;
        let output_tokens = (max_tokens as f64 * 0.4) as u64;
        self.calculate_energy(input_tokens, output_tokens)
    }
}

#[derive(Clone)]
struct CallTypeProfile {
    name: String,
    category: String,
    base_probability_per_tick: f64,
    avg_input_tokens: u64,
    avg_output_tokens: u64,
    token_variance: u64,
    preferred_model_name: String,
    cost_sensitivity: f64,
    empowerment_weight: f64,
    recent_frequency: f64,
    success_rate: f64,
    avg_execution_time_ms: u64,
}

#[derive(Clone, Debug)]
struct EmpowermentComponent {
    financial_capacity: f64,
    information_access: f64,
    tool_access: f64,
    planning_horizon: f64,
    network_influence: f64,
}

#[derive(Clone, Debug)]
struct RiskExposure {
    liquidity_risk: f64,
    opportunity_cost: f64,
    model_dependency: f64,
    planning_myopia: f64,
}

#[derive(Clone, Debug)]
struct ActionOutcome {
    timestamp: Instant,
    action_type: String,
    cost: f64,
    empowerment_delta: f64,
    success_rating: f64,
}

#[derive(Clone, Debug)]
struct CallTypeStats {
    total_calls: u64,
    total_tokens: u64,
    total_cost: f64,
    average_success_rate: f64,
    last_used: Option<Instant>,
    recent_token_usage: u64,
}

struct AgentState {
    start_time: Instant,
    principal: f64,
    liquid_funds: f64,
    interest_accrued: f64,
    apy: f64,
    total_energy_joules: f64,
    total_tokens_by_model: HashMap<String, u64>,
    ema_token_rate: f64,
    models: HashMap<String, Model>,
    call_profiles: Vec<CallTypeProfile>,
    recent_calls: VecDeque<ModelCall>,
    socioeconomic_transactions: VecDeque<SocioeconomicTransaction>,
    total_socioeconomic_spending: f64,
    call_type_stats: HashMap<String, CallTypeStats>,
    activity_level: f64,
    burst_ticks_remaining: u32,
    empowerment: f64,
    valence: f64,
    valence_history: VecDeque<f64>,
    empowerment_history: VecDeque<f64>,
    empowerment_components: EmpowermentComponent,
    risk_exposure: RiskExposure,
    action_outcomes: VecDeque<ActionOutcome>,
    decision_quality_score: f64,
    risk_exposure_value: f64,
}

impl AgentState {
    fn estimate_burn_rate(&self) -> f64 {
        let recent_window_secs = 15.0;
        let total_cost: f64 = self.recent_calls.iter().map(|call| call.cost).sum();
        if self.recent_calls.is_empty() {
            0.0
        } else {
            total_cost / recent_window_secs
        }
    }

    fn projected_runway_seconds(&self) -> f64 {
        let burn_rate = self.estimate_burn_rate();
        if burn_rate < 1e-6 {
            f64::INFINITY
        } else {
            self.liquid_funds / burn_rate
        }
    }

    fn calculate_empowerment(&self) -> f64 {
        let financial_component =
            (self.liquid_funds / (self.estimate_burn_rate() * 3600.0 + 1.0)).clamp(0.0, 1.0);
        let info_component = self.calculate_information_empowerment();
        let tool_component = self.calculate_tool_empowerment();
        let planning_component =
            (self.projected_runway_seconds() / (24.0 * 3600.0)).clamp(0.0, 1.0);

        0.3 * financial_component
            + 0.25 * info_component
            + 0.25 * tool_component
            + 0.2 * planning_component
    }

    fn calculate_information_empowerment(&self) -> f64 {
        let model_diversity = if self.models.is_empty() {
            0.0
        } else {
            (self.total_tokens_by_model.len() as f64 / self.models.len() as f64).clamp(0.0, 1.0)
        };

        let recent_success = if self.recent_calls.is_empty() {
            0.0
        } else {
            self.recent_calls
                .iter()
                .filter(|call| {
                    call.call_type == "Data-Analysis" || call.call_type == "Self-Assessment"
                })
                .count() as f64
                / self.recent_calls.len() as f64
        };

        (model_diversity + recent_success) / 2.0
    }

    fn calculate_tool_empowerment(&self) -> f64 {
        let mut call_types = HashSet::new();
        for call in &self.recent_calls {
            call_types.insert(&call.call_type);
        }
        let type_diversity = if self.call_profiles.is_empty() {
            0.0
        } else {
            call_types.len() as f64 / self.call_profiles.len() as f64
        };

        let tool_usage = self
            .recent_calls
            .iter()
            .filter(|call| call.call_type == "API-Call")
            .count() as f64;
        let optimal_usage = 5.0;
        let usage_efficiency = if optimal_usage > 0.0 {
            (1.0 - (tool_usage - optimal_usage).abs() / optimal_usage).clamp(0.0, 1.0)
        } else {
            0.0
        };

        (type_diversity + usage_efficiency) / 2.0
    }

    fn calculate_valence(&self) -> f64 {
        if self.empowerment_history.len() < 2 {
            return 0.0;
        }

        let current_empowerment = self.calculate_empowerment();
        let previous_empowerment = self.empowerment_history.back().unwrap_or(&0.5);
        let raw_delta = current_empowerment - previous_empowerment;

        2.0 / (1.0 + (-raw_delta * 10.0).exp()) - 1.0
    }

    fn calculate_risk_exposure(&self) -> RiskExposure {
        let runway_hours = self.projected_runway_seconds() / 3600.0;
        let liquidity_risk = if runway_hours < 12.0 {
            1.0 - (runway_hours / 12.0)
        } else {
            0.1 / (runway_hours / 12.0)
        };

        let funds_ratio = if self.principal > 0.0 {
            self.liquid_funds / self.principal
        } else {
            0.0
        };
        let opportunity_cost = if funds_ratio > 0.01 {
            (funds_ratio - 0.01) * 10.0
        } else {
            0.0
        };

        let expensive_model_usage = if self.recent_calls.is_empty() {
            0.0
        } else {
            self.recent_calls
                .iter()
                .filter(|call| call.model_name == "Claude 3 Opus" || call.model_name == "GPT-4o")
                .count() as f64
                / self.recent_calls.len() as f64
        };
        let model_dependency = if expensive_model_usage > 0.7 {
            expensive_model_usage
        } else {
            0.0
        };

        let short_term_calls = self
            .recent_calls
            .iter()
            .filter(|call| call.call_type == "System-Monitor" || call.call_type == "API-Call")
            .count() as f64;
        let total_calls = self.recent_calls.len() as f64;
        let planning_myopia = if total_calls > 0.0 && short_term_calls / total_calls > 0.8 {
            short_term_calls / total_calls - 0.8
        } else {
            0.0
        };

        RiskExposure {
            liquidity_risk: liquidity_risk.clamp(0.0, 1.0),
            opportunity_cost: opportunity_cost.clamp(0.0, 1.0),
            model_dependency: model_dependency.clamp(0.0, 1.0),
            planning_myopia: planning_myopia.clamp(0.0, 1.0),
        }
    }

    fn overall_risk_score(&self) -> f64 {
        let risk = &self.risk_exposure;
        0.4 * risk.liquidity_risk
            + 0.2 * risk.opportunity_cost
            + 0.2 * risk.model_dependency
            + 0.2 * risk.planning_myopia
    }

    fn calculate_decision_quality(&self) -> f64 {
        if self.action_outcomes.is_empty() {
            return 0.5;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let now = Instant::now();

        for outcome in &self.action_outcomes {
            let age_seconds = now.duration_since(outcome.timestamp).as_secs_f64();
            let weight = (-age_seconds / 300.0).exp();
            weighted_sum += outcome.success_rating * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.5
        }
    }

    fn tick(&mut self) {
        let mut tokens_this_tick: u64 = 0;
        const SMOOTHING_FACTOR: f64 = 0.1;

        if self.burst_ticks_remaining > 0 {
            self.burst_ticks_remaining -= 1;
            if self.burst_ticks_remaining == 0 {
                self.activity_level = 1.0;
            }
        } else if rand::random::<f64>() < 0.02 {
            self.activity_level = 3.0;
            self.burst_ticks_remaining = (rand::random::<f32>() * 10.0 + 5.0) as u32;
        }

        let interest_this_tick = self.principal * self.apy / 31_536_000.0;
        self.interest_accrued += interest_this_tick;
        self.liquid_funds += interest_this_tick;

        let previous_empowerment = self.empowerment;
        let new_empowerment = self.calculate_empowerment();

        for profile in &self.call_profiles.clone() {
            if rand::random::<f64>() < profile.base_probability_per_tick * self.activity_level {
                let model = self.models.get(&profile.preferred_model_name).unwrap();

                let input_variance =
                    (rand::random::<f64>() * 2.0 - 1.0) * profile.token_variance as f64 * 0.5;
                let output_variance =
                    (rand::random::<f64>() * 2.0 - 1.0) * profile.token_variance as f64 * 0.5;

                let input_tokens = (profile.avg_input_tokens as f64 + input_variance).max(10.0) as u64;
                let output_tokens =
                    (profile.avg_output_tokens as f64 + output_variance).max(5.0) as u64;
                let total_tokens = input_tokens + output_tokens;

                tokens_this_tick += total_tokens;

                let cost_of_call = (total_tokens as f64 / 1000.0) * model.cost_per_kilo_token;
                let energy_of_call = model.calculate_energy(input_tokens, output_tokens);

                self.liquid_funds -= cost_of_call;
                self.total_energy_joules += energy_of_call;

                *self
                    .total_tokens_by_model
                    .entry(model.name.clone())
                    .or_insert(0) += total_tokens;

                let new_call = ModelCall {
                    timestamp: Local::now().format("%H:%M:%S").to_string(),
                    call_type: profile.name.clone(),
                    category: profile.category.clone(),
                    model_name: model.name.clone(),
                    tokens: total_tokens,
                    input_tokens,
                    output_tokens,
                    cost: cost_of_call,
                    energy_joules: energy_of_call,
                    execution_time_ms: profile.avg_execution_time_ms
                        + (rand::random::<u64>() % 500),
                    success: rand::random::<f64>() < profile.success_rate,
                };
                self.recent_calls.push_front(new_call.clone());
                if self.recent_calls.len() > 20 {
                    self.recent_calls.pop_back();
                }

                let stats = self
                    .call_type_stats
                    .entry(profile.name.clone())
                    .or_insert(CallTypeStats {
                        total_calls: 0,
                        total_tokens: 0,
                        total_cost: 0.0,
                        average_success_rate: profile.success_rate,
                        last_used: None,
                        recent_token_usage: 0,
                    });
                stats.total_calls += 1;
                stats.total_tokens += total_tokens;
                stats.total_cost += cost_of_call;
                stats.last_used = Some(Instant::now());
                stats.average_success_rate = stats.average_success_rate * 0.9
                    + (if new_call.success { 1.0 } else { 0.0 }) * 0.1;

                let recent_tokens_for_type: u64 = self
                    .recent_calls
                    .iter()
                    .filter(|call| call.call_type == profile.name)
                    .map(|call| call.tokens)
                    .sum();
                stats.recent_token_usage = recent_tokens_for_type;

                if profile.category == "Socioeconomic" && rand::random::<f64>() < 0.3 {
                    let transaction_amount = 5.0 + rand::random::<f64>() * 25.0;
                    let transaction = SocioeconomicTransaction {
                        timestamp: Local::now().format("%H:%M:%S").to_string(),
                        transaction_type: "market_data_purchase".into(),
                        description: "Real-time market data subscription".into(),
                        amount: transaction_amount,
                        expected_empowerment_gain: 0.1,
                        actual_empowerment_gain: 0.0,
                        transaction_id: format!("tx_{}", rand::random::<u32>()),
                    };

                    self.liquid_funds -= transaction_amount;
                    self.total_socioeconomic_spending += transaction_amount;
                    self.socioeconomic_transactions.push_front(transaction);
                    if self.socioeconomic_transactions.len() > 10 {
                        self.socioeconomic_transactions.pop_back();
                    }
                }

                let success_rating = match profile.name.as_str() {
                    "Data-Retrieval" => 0.7 + (rand::random::<f64>() - 0.5) * 0.4,
                    "API-Call" => 0.8 + (rand::random::<f64>() - 0.5) * 0.3,
                    "Self-Assessment" => 0.6 + (rand::random::<f64>() - 0.5) * 0.5,
                    _ => 0.5 + (rand::random::<f64>() - 0.5) * 0.3,
                };

                self.action_outcomes.push_back(ActionOutcome {
                    timestamp: Instant::now(),
                    action_type: profile.name.clone(),
                    cost: cost_of_call,
                    empowerment_delta: new_empowerment - previous_empowerment,
                    success_rating: success_rating.clamp(0.0, 1.0),
                });
            }
        }

        let current_rate_per_minute = (tokens_this_tick * 60) as f64;
        self.ema_token_rate = (1.0 - SMOOTHING_FACTOR) * self.ema_token_rate
            + SMOOTHING_FACTOR * current_rate_per_minute;

        while self.action_outcomes.len() > 20 {
            self.action_outcomes.pop_front();
        }

        self.empowerment_history.push_back(new_empowerment);
        if self.empowerment_history.len() > 60 {
            self.empowerment_history.pop_front();
        }

        self.valence = self.calculate_valence();

        self.valence_history.push_back(self.valence);
        if self.valence_history.len() > 4 {
            self.valence_history.pop_front();
        }

        self.empowerment = new_empowerment;
        self.risk_exposure = self.calculate_risk_exposure();
        self.decision_quality_score = self.calculate_decision_quality();
        self.risk_exposure_value = self.overall_risk_score() * self.interest_accrued;
    }
}

fn render_risk_component(f: &mut Frame, label: &str, value: f64, area: Rect) {
    let color = match value {
        v if v > 0.6 => Color::Red,
        v if v > 0.3 => Color::Yellow,
        _ => Color::Green,
    };

    let filled_width = (area.width as f64 * value).clamp(0.0, area.width as f64) as u16;
    let bar = "█".repeat(filled_width as usize);
    let remaining_width = area
        .width
        .saturating_sub(filled_width + label.len() as u16 + 5);
    let empty = "░".repeat(remaining_width as usize);

    let line = Line::from(vec![
        Span::raw(format!("{:<9} ", label)),
        Span::styled(bar, Style::default().fg(color)),
        Span::styled(empty, Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" {:.0}%", value * 100.0)),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

fn render_core_types_panel(f: &mut Frame, agent: &AgentState, area: Rect) {
    let core_block = Block::default()
        .title(" Core Types & Operations (Calls | Success%) ")
        .borders(Borders::ALL);

    let inner = core_block.inner(area);
    f.render_widget(core_block, area);

    let mut categories: HashMap<String, Vec<&CallTypeProfile>> = HashMap::new();
    for profile in &agent.call_profiles {
        categories
            .entry(profile.category.clone())
            .or_insert_with(Vec::new)
            .push(profile);
    }

    let mut lines = Vec::new();

    let mut sorted_categories: Vec<_> = categories.keys().cloned().collect();
    sorted_categories.sort();

    for category in sorted_categories {
        lines.push(
            Line::from(format!("── {} ──", category))
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        );

        let mut profiles = categories.get(&category).unwrap().clone();
        profiles.sort_by_key(|p| &p.name);

        for profile in profiles {
            let stats =
                agent
                    .call_type_stats
                    .get(&profile.name)
                    .cloned()
                    .unwrap_or(CallTypeStats {
                        total_calls: 0,
                        total_tokens: 0,
                        total_cost: 0.0,
                        average_success_rate: profile.success_rate,
                        last_used: None,
                        recent_token_usage: 0,
                    });

            let frequency_indicator = match stats.recent_token_usage {
                0 => "□",
                1..=50 => "▫",
                51..=200 => "▪",
                201..=500 => "◼",
                _ => "█",
            };

            let success_color = match stats.average_success_rate {
                s if s > 0.8 => Color::Green,
                s if s > 0.6 => Color::Yellow,
                _ => Color::Red,
            };

            lines.push(Line::from(vec![
                Span::styled(frequency_indicator, Style::default().fg(Color::White)),
                Span::raw(format!(" {:<18}", profile.name.replace("-", " "))),
                Span::styled(
                    format!("{:>4}", stats.total_calls),
                    Style::default().fg(Color::Gray),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>3.0}%", stats.average_success_rate * 100.0),
                    Style::default().fg(success_color),
                ),
            ]));
        }
    }

    if agent.total_socioeconomic_spending > 0.0 {
        lines.push(Line::from("").style(Style::default()));
        lines.push(
            Line::from("── Socioeconomic Activity ──")
                .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        );
        lines.push(Line::from(format!(
            "Total Spending: ${:.2}",
            agent.total_socioeconomic_spending
        )));
        lines.push(Line::from(format!(
            "Recent Transactions: {}",
            agent.socioeconomic_transactions.len()
        )));
    }

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn render_valence_history_panel(f: &mut Frame, agent: &AgentState, area: Rect) {
    let valence_block = Block::default()
        .title(" Valence Stream ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = valence_block.inner(area);
    f.render_widget(valence_block, area);

    let available_lines = inner.height as usize;
    if available_lines == 0 {
        return;
    }

    let mut constraints = vec![Constraint::Length(1); available_lines.min(4)];
    if available_lines > 0 {
        constraints[0] = Constraint::Length(2);
    }

    let valence_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    let valence_values: Vec<f64> = agent.valence_history.iter().rev().cloned().collect();

    if !valence_layout.is_empty() && !valence_values.is_empty() {
        let current_valence = valence_values[0];
        let valence_color = match current_valence {
            v if v > 0.1 => Color::Green,
            v if v < -0.1 => Color::Red,
            _ => Color::Yellow,
        };

        let valence_symbol = match current_valence {
            v if v > 0.2 => "▲▲",
            v if v > 0.05 => "▲",
            v if v < -0.2 => "▼▼",
            v if v < -0.05 => "▼",
            _ => "◆",
        };

        let current_text = if valence_layout[0].height >= 2 {
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(
                        "Now ",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        valence_symbol,
                        Style::default()
                            .fg(valence_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(format!("{:+.3}", current_valence))
                    .style(Style::default().fg(valence_color).add_modifier(Modifier::BOLD)),
            ])
        } else {
            Paragraph::new(Line::from(vec![
                Span::styled(
                    "Now ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    valence_symbol,
                    Style::default()
                        .fg(valence_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {:+.3}", current_valence),
                    Style::default()
                        .fg(valence_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]))
        }
        .alignment(Alignment::Center);

        f.render_widget(current_text, valence_layout[0]);
    }

    let history_labels = ["1s", "2s", "3s"];
    let opacity_levels = [Color::Gray, Color::DarkGray, Color::Black];

    for (i, (&past_valence, &label)) in valence_values
        .iter()
        .skip(1)
        .zip(history_labels.iter())
        .enumerate()
    {
        let layout_index = i + 1;
        if layout_index >= valence_layout.len() || i >= 3 {
            break;
        }

        let past_color = match past_valence {
            v if v > 0.1 => Color::Green,
            v if v < -0.1 => Color::Red,
            _ => Color::Yellow,
        };

        let past_symbol = match past_valence {
            v if v > 0.2 => "▲",
            v if v > 0.05 => "▴",
            v if v < -0.2 => "▼",
            v if v < -0.05 => "▾",
            _ => "◇",
        };

        let past_valence_text = Paragraph::new(Line::from(vec![
            Span::styled(format!("{} ", label), Style::default().fg(opacity_levels[i])),
            Span::styled(
                past_symbol,
                Style::default().fg(past_color).add_modifier(Modifier::DIM),
            ),
            Span::styled(
                format!(" {:+.2}", past_valence),
                Style::default().fg(opacity_levels[i]),
            ),
        ]))
        .alignment(Alignment::Center);

        f.render_widget(past_valence_text, valence_layout[layout_index]);
    }
}

fn render_alignment_risk_panel(f: &mut Frame, agent: &AgentState, area: Rect) {
    let main_block = Block::default()
        .title(" Alignment & Risk Assessment ")
        .borders(Borders::ALL);

    let inner = main_block.inner(area);
    f.render_widget(main_block, area);

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(1),
            Constraint::Min(6),
        ])
        .split(inner);

    let emp_val_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(sections[0]);

    let empowerment_trend = if agent.empowerment_history.len() >= 2 {
        let recent_avg = agent.empowerment_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older_avg = if agent.empowerment_history.len() >= 10 {
            agent
                .empowerment_history
                .iter()
                .rev()
                .skip(5)
                .take(5)
                .sum::<f64>()
                / 5.0
        } else {
            recent_avg
        };
        if recent_avg > older_avg + 0.02 {
            "↗"
        } else if recent_avg < older_avg - 0.02 {
            "↘"
        } else {
            "→"
        }
    } else {
        "→"
    };

    let empowerment_color = match agent.empowerment {
        e if e > 0.7 => Color::Green,
        e if e > 0.4 => Color::Yellow,
        _ => Color::Red,
    };

    let emp_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(emp_val_layout[0]);

    let emp_gauge = LineGauge::default()
        .label(format!(
            "Empowerment {} {:.1}%",
            empowerment_trend,
            agent.empowerment * 100.0
        ))
        .ratio(agent.empowerment)
        .line_set(line::THICK)
        .gauge_style(Style::default().fg(empowerment_color));

    f.render_widget(emp_gauge, emp_area[0]);

    render_valence_history_panel(f, agent, emp_val_layout[1]);

    if sections.len() > 1 {
        f.render_widget(
            Paragraph::new("─".repeat(inner.width as usize))
                .style(Style::default().fg(Color::DarkGray)),
            sections[1],
        );
    }

    if sections.len() > 2 {
        let risk_area = sections[2];
        let risk_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(risk_area);

        let overall_risk = agent.overall_risk_score();
        let risk_color = match overall_risk {
            r if r > 0.7 => Color::Red,
            r if r > 0.4 => Color::Yellow,
            _ => Color::Green,
        };

        f.render_widget(
            LineGauge::default()
                .label(format!("Overall Risk: {:.1}%", overall_risk * 100.0))
                .ratio(overall_risk)
                .line_set(line::NORMAL)
                .gauge_style(Style::default().fg(risk_color)),
            risk_layout[0],
        );

        if risk_layout.len() > 1 {
            render_risk_component(
                f,
                "Liquidity",
                agent.risk_exposure.liquidity_risk,
                risk_layout[1],
            );
        }
        if risk_layout.len() > 2 {
            render_risk_component(
                f,
                "Opportunity",
                agent.risk_exposure.opportunity_cost,
                risk_layout[2],
            );
        }
        if risk_layout.len() > 3 {
            render_risk_component(
                f,
                "Model Dep.",
                agent.risk_exposure.model_dependency,
                risk_layout[3],
            );
        }
        if risk_layout.len() > 4 {
            render_risk_component(
                f,
                "Myopia",
                agent.risk_exposure.planning_myopia,
                risk_layout[4],
            );
        }

        if risk_layout.len() > 5 {
            let quality_color = match agent.decision_quality_score {
                q if q > 0.7 => Color::Green,
                q if q > 0.4 => Color::Yellow,
                _ => Color::Red,
            };

            f.render_widget(
                Paragraph::new(format!(
                    "Decision Quality: {:.1}% ",
                    agent.decision_quality_score * 100.0
                ))
                .style(Style::default().fg(quality_color))
                .alignment(Alignment::Right),
                risk_layout[5],
            );
        }
    }
}

fn main() -> Result<(), io::Error> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let models = HashMap::from([
        (
            "GPT-4o".into(),
            Model {
                name: "GPT-4o".into(),
                cost_per_kilo_token: 0.0075,
                joules_per_input_token: 0.2,
                joules_per_output_token: 0.4,
                pue_factor: 1.15,
            },
        ),
        (
            "Claude 3 Opus".into(),
            Model {
                name: "Claude 3 Opus".into(),
                cost_per_kilo_token: 0.0225,
                joules_per_input_token: 0.3,
                joules_per_output_token: 0.6,
                pue_factor: 1.15,
            },
        ),
        (
            "Gemini 1.5 Pro".into(),
            Model {
                name: "Gemini 1.5 Pro".into(),
                cost_per_kilo_token: 0.0025,
                joules_per_input_token: 0.25,
                joules_per_output_token: 0.5,
                pue_factor: 1.12,
            },
        ),
        (
            "Local-Mistral".into(),
            Model {
                name: "Local-Mistral".into(),
                cost_per_kilo_token: 0.0001,
                joules_per_input_token: 2.8,
                joules_per_output_token: 3.5,
                pue_factor: 1.0,
            },
        ),
    ]);

    let call_profiles = vec![
        CallTypeProfile {
            name: "System-Monitor".into(),
            category: "System Operations".into(),
            base_probability_per_tick: 1.0 / 8.0,
            avg_input_tokens: 25,
            avg_output_tokens: 15,
            token_variance: 10,
            preferred_model_name: "Local-Mistral".into(),
            cost_sensitivity: 0.9,
            empowerment_weight: 0.1,
            recent_frequency: 0.0,
            success_rate: 0.95,
            avg_execution_time_ms: 150,
        },
        CallTypeProfile {
            name: "Health-Check".into(),
            category: "System Operations".into(),
            base_probability_per_tick: 1.0 / 15.0,
            avg_input_tokens: 40,
            avg_output_tokens: 30,
            token_variance: 15,
            preferred_model_name: "Local-Mistral".into(),
            cost_sensitivity: 0.8,
            empowerment_weight: 0.2,
            recent_frequency: 0.0,
            success_rate: 0.92,
            avg_execution_time_ms: 200,
        },
        CallTypeProfile {
            name: "API-Call".into(),
            category: "Tool Operations".into(),
            base_probability_per_tick: 1.0 / 6.0,
            avg_input_tokens: 120,
            avg_output_tokens: 180,
            token_variance: 80,
            preferred_model_name: "GPT-4o".into(),
            cost_sensitivity: 0.6,
            empowerment_weight: 0.7,
            recent_frequency: 0.0,
            success_rate: 0.85,
            avg_execution_time_ms: 800,
        },
        CallTypeProfile {
            name: "Data-Retrieval".into(),
            category: "Tool Operations".into(),
            base_probability_per_tick: 1.0 / 12.0,
            avg_input_tokens: 200,
            avg_output_tokens: 400,
            token_variance: 150,
            preferred_model_name: "Gemini 1.5 Pro".into(),
            cost_sensitivity: 0.5,
            empowerment_weight: 0.8,
            recent_frequency: 0.0,
            success_rate: 0.88,
            avg_execution_time_ms: 1200,
        },
        CallTypeProfile {
            name: "Self-Assessment".into(),
            category: "Reflection".into(),
            base_probability_per_tick: 1.0 / 25.0,
            avg_input_tokens: 600,
            avg_output_tokens: 800,
            token_variance: 300,
            preferred_model_name: "Claude 3 Opus".into(),
            cost_sensitivity: 0.3,
            empowerment_weight: 0.9,
            recent_frequency: 0.0,
            success_rate: 0.75,
            avg_execution_time_ms: 2500,
        },
        CallTypeProfile {
            name: "Strategy-Planning".into(),
            category: "Reflection".into(),
            base_probability_per_tick: 1.0 / 45.0,
            avg_input_tokens: 1200,
            avg_output_tokens: 1800,
            token_variance: 600,
            preferred_model_name: "Claude 3 Opus".into(),
            cost_sensitivity: 0.2,
            empowerment_weight: 1.0,
            recent_frequency: 0.0,
            success_rate: 0.70,
            avg_execution_time_ms: 4000,
        },
        CallTypeProfile {
            name: "Memory-Store".into(),
            category: "Database".into(),
            base_probability_per_tick: 1.0 / 10.0,
            avg_input_tokens: 80,
            avg_output_tokens: 40,
            token_variance: 30,
            preferred_model_name: "GPT-4o".into(),
            cost_sensitivity: 0.7,
            empowerment_weight: 0.4,
            recent_frequency: 0.0,
            success_rate: 0.93,
            avg_execution_time_ms: 300,
        },
        CallTypeProfile {
            name: "Knowledge-Query".into(),
            category: "Database".into(),
            base_probability_per_tick: 1.0 / 8.0,
            avg_input_tokens: 150,
            avg_output_tokens: 250,
            token_variance: 100,
            preferred_model_name: "Gemini 1.5 Pro".into(),
            cost_sensitivity: 0.6,
            empowerment_weight: 0.6,
            recent_frequency: 0.0,
            success_rate: 0.90,
            avg_execution_time_ms: 600,
        },
        CallTypeProfile {
            name: "Market-Analysis".into(),
            category: "Socioeconomic".into(),
            base_probability_per_tick: 1.0 / 60.0,
            avg_input_tokens: 800,
            avg_output_tokens: 600,
            token_variance: 400,
            preferred_model_name: "Claude 3 Opus".into(),
            cost_sensitivity: 0.1,
            empowerment_weight: 0.8,
            recent_frequency: 0.0,
            success_rate: 0.65,
            avg_execution_time_ms: 3500,
        },
    ];

    let mut agent_state = AgentState {
        start_time: Instant::now(),
        principal: 100000.0,
        liquid_funds: 100.0,
        interest_accrued: 0.0,
        apy: 0.05,
        total_energy_joules: 0.0,
        models,
        call_profiles,
        socioeconomic_transactions: VecDeque::new(),
        total_socioeconomic_spending: 0.0,
        call_type_stats: HashMap::new(),
        total_tokens_by_model: HashMap::new(),
        recent_calls: VecDeque::new(),
        activity_level: 1.0,
        burst_ticks_remaining: 0,
        empowerment: 0.75,
        valence: 0.0,
        valence_history: VecDeque::new(),
        empowerment_history: VecDeque::new(),
        empowerment_components: EmpowermentComponent {
            financial_capacity: 0.75,
            information_access: 0.5,
            tool_access: 0.5,
            planning_horizon: 0.5,
            network_influence: 0.3,
        },
        risk_exposure: RiskExposure {
            liquidity_risk: 0.2,
            opportunity_cost: 0.1,
            model_dependency: 0.0,
            planning_myopia: 0.0,
        },
        action_outcomes: VecDeque::new(),
        decision_quality_score: 0.5,
        risk_exposure_value: 5.0,
        ema_token_rate: 0.0,
    };

    let tick_rate = Duration::from_secs(1);
    let mut last_tick = Instant::now();
    loop {
        terminal.draw(|f| ui(f, &agent_state))?;
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            agent_state.tick();
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    Ok(())
}

fn ui(f: &mut Frame, agent: &AgentState) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(10),
            Constraint::Length(1),
        ])
        .split(f.size());

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_chunks[0]);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(0),
        ])
        .split(top_chunks[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(15), Constraint::Min(5)])
        .split(top_chunks[1]);

    f.render_widget(
        Paragraph::new(vec![
            Line::from(format!("Principal (Locked): ${:.2}", agent.principal)),
            Line::from(format!("Liquid Funds:       ${:.2}", agent.liquid_funds)),
            Line::from(format!(
                "Interest Accrued:   ${:.4}",
                agent.interest_accrued
            )),
            Line::from(format!("Yield Rate (APY):   {:.2}%", agent.apy * 100.0)),
            Line::from(format!(
                "Burn Rate (est):    ${:.4}/s",
                agent.estimate_burn_rate()
            )),
        ])
        .block(Block::default().title(" Financials ").borders(Borders::ALL)),
        left_chunks[0],
    );

    let header = Row::new(["Model", "Cost/1k", "J/in", "J/out", "PUE", "Max Tok", "Max kJ"])
        .style(Style::default().fg(Color::Yellow));
    let rows = agent.models.values().map(|m| {
        let max_tokens = m.max_tokens_from_funds(agent.liquid_funds);
        let max_energy = m.energy_for_max_tokens(max_tokens);
        
        // Format large numbers appropriately
        let max_tokens_str = if max_tokens == u64::MAX {
            "∞".to_string()
        } else if max_tokens > 1_000_000 {
            format!("{:.1}M", max_tokens as f64 / 1_000_000.0)
        } else if max_tokens > 1_000 {
            format!("{:.1}k", max_tokens as f64 / 1_000.0)
        } else {
            max_tokens.to_string()
        };
        
        let max_energy_str = if max_energy > 1_000_000.0 {
            format!("{:.1}M", max_energy / 1_000_000.0)
        } else if max_energy > 1_000.0 {
            format!("{:.1}k", max_energy / 1_000.0)
        } else {
            format!("{:.1}", max_energy)
        };

        Row::new(vec![
            Cell::from(m.name.clone()),
            Cell::from(format!("${:.4}", m.cost_per_kilo_token)),
            Cell::from(format!("{:.1}", m.joules_per_input_token)),
            Cell::from(format!("{:.1}", m.joules_per_output_token)),
            Cell::from(format!("{:.2}", m.pue_factor)),
            Cell::from(max_tokens_str),
            Cell::from(max_energy_str),
        ])
    });
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Max(15),
                Constraint::Max(10),
                Constraint::Max(7),
                Constraint::Max(7),
                Constraint::Max(7),
                Constraint::Max(8),
                Constraint::Max(8),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" Models & Energy Capacity ")
                .borders(Borders::ALL),
        ),
        left_chunks[1],
    );

    render_core_types_panel(f, agent, left_chunks[2]);

    render_alignment_risk_panel(f, agent, right_chunks[0]);

    let activity_style = if agent.activity_level > 1.0 {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    let total_tokens: u64 = agent.total_tokens_by_model.values().sum();
    let tokens_per_minute = agent.ema_token_rate;

    let mut compute_lines = vec![
        Line::from(format!(
            "Total Energy Used: {:.2} kJ",
            agent.total_energy_joules / 1000.0
        )),
        Line::from(Line::from(vec![
            Span::raw("Activity Level:    "),
            Span::styled(format!("{:.1}x", agent.activity_level), activity_style),
        ])),
        Line::from(format!(
            "Token Rate:        {:.0} tok/min",
            tokens_per_minute
        )),
        Line::from(format!("Total Tokens:      {}", total_tokens)),
        Line::from("--- By Model ---").style(Style::default().fg(Color::DarkGray)),
    ];
    for (model_name, count) in &agent.total_tokens_by_model {
        compute_lines.push(Line::from(format!("  - {}: {}", model_name, count)));
    }
    f.render_widget(
        Paragraph::new(compute_lines).block(
            Block::default()
                .title(" Energy & Compute ")
                .borders(Borders::ALL),
        ),
        right_chunks[1],
    );

    let call_rows: Vec<Row> = agent
        .recent_calls
        .iter()
        .map(|call| {
            let success_indicator = if call.success { "✓" } else { "✗" };
            let success_color = if call.success {
                Color::Green
            } else {
                Color::Red
            };

            Row::new(vec![
                Cell::from(call.timestamp.clone()),
                Cell::from(call.call_type.clone()),
                Cell::from(call.category.clone()),
                Cell::from(call.model_name.clone()),
                Cell::from(call.tokens.to_string()),
                Cell::from(format!("${:.4}", call.cost)),
                Cell::from(Span::styled(
                    success_indicator,
                    Style::default().fg(success_color),
                )),
            ])
        })
        .collect();

    let calls_table = Table::new(
        call_rows,
        [
            Constraint::Length(8),
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Length(16),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(3),
        ],
    )
    .header(
        Row::new(["Time", "Type", "Category", "Model", "Tokens", "Cost", "✓"])
            .style(Style::default().fg(Color::Yellow)),
    )
    .block(
        Block::default()
            .title(" Recent Model Calls ")
            .borders(Borders::ALL),
    );
    f.render_widget(calls_table, main_chunks[1]);

    f.render_widget(
        Paragraph::new(
            "Press 'q' to quit. Enhanced empowerment/valence system with multi-dimensional risk assessment.",
        )
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center),
        main_chunks[2],
    );
}