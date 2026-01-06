# Kalman-Filtered Trend Trader: A Deep Reinforcement Learning Agent for Portfolio Optimization – Weeks 1 & 2

---

## Overview

This repository documents the work completed during the first two weeks of the WiDS 2025 project(till now). The focus was on building strong foundations in statistical learning and then applying them to a real-world financial time-series problem. These weeks lay the groundwork for later reinforcement learning–based decision making.

---

## Week 1: Statistical Learning & Bias Analysis

**Key Topics**

* Multiple Linear Regression and OLS derivation
* Assumptions: linearity, independence, homoscedasticity, normality, multicollinearity
* Bias–Variance decomposition

**Implementation & Analysis**

* Residuals vs Fitted plots and Q–Q plots for diagnostics
* Leverage and Cook’s Distance for influential points
* Empirical study of multicollinearity and variance inflation

**Fairness Case Study (Salary Prediction)**

* Data preprocessing with imputation and one-hot encoding
* Gender-stratified train–test split
* Fairness metrics: MAE by group, Demographic Parity, Equal Opportunity, Disparate Impact
* SHAP analysis showing bias linked to gender and education

---

## Week 2: Time-Series Modeling & Trading Strategy

**Problem Setup**

* Asset: MSFT daily prices (2015–2024)
* Objective: predict next-day price movement and generate trading signals

**Feature Engineering**

* Moving averages (20d, 60d), volatility, momentum spread
* 60-day Rate of Change (roc_60) for regime stability

**Kalman Filter**

* State-space model to estimate time-varying momentum (βₜ)
* Kalman states and prediction errors used as ML features

**Machine Learning & Trading Logic**

* Linear regression to predict next-day price ratio (Pₜ₊₁ / Pₜ)
* Signals: Long / Short / Neutral using a threshold ε (0.002)
* Transaction costs modeled (0.1%)


---

## Summary

* Week 1 built theoretical and empirical understanding of regression, diagnostics, and fairness
* Week 2 introduced dynamic time-series modeling and ML-based trading decisions
* Together, these weeks establish the analytical foundation for future RL-based extensions
