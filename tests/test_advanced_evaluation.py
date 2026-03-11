"""
高级评估框架测试模块。

测试四大评估维度的功能：
1. 交易质量分析（MAE/MFE）
2. 市场状态切割
3. 可解释性归因
4. 跨市场分析
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.evaluation.trade_analyzer import (
    TradeQualityMetrics,
    TradeQualitySummary,
    TradeAnalyzer,
)

from src.evaluation.market_state_analyzer import (
    MarketState,
    MarketStateMetrics,
    MarketStateSummary,
    MarketStateAnalyzer,
)

from src.evaluation.ml_explainer import (
    FeatureAttribution,
    MLExplanationResult,
    MLExplainer,
)

from src.evaluation.llm_explainer import (
    ThemeAttribution,
    LLMExplanationResult,
    LLMExplainer,
)

from src.evaluation.cross_market_analyzer import (
    SignalDecayResult,
    CrossMarketSummary,
    CrossMarketAnalyzer,
)

from src.evaluation.attribution_comparator import (
    AttributionAlignment,
    ComparisonResult,
    AttributionComparator,
)

from src.evaluation.advanced_visualizer import (
    VisualizationConfig,
    AdvancedVisualizer,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def sample_ohlcv():
    """创建示例 OHLCV 数据。"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

    base_price = 100
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        "open": prices * (1 + np.random.randn(100) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)


@pytest.fixture
def sample_trade_log():
    """创建示例交易日志。"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-10", periods=20, freq="5B")

    return pd.DataFrame({
        "date": dates,
        "type": ["买入", "卖出"] * 10,
        "price": 100 + np.cumsum(np.random.randn(20) * 2),
        "size": [100] * 20,
        "commission": [10] * 20,
    })


@pytest.fixture
def sample_vix_data():
    """创建示例 VIX 数据。"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    vix_values = 15 + np.cumsum(np.random.randn(100) * 0.5)
    vix_values = np.clip(vix_values, 10, 50)

    return pd.DataFrame({"vix": vix_values}, index=dates)


@pytest.fixture
def sample_equity_curve():
    """创建示例净值曲线。"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    nav = 1.0 * (1 + np.random.randn(100) * 0.01 + 0.0003).cumprod()

    return pd.DataFrame({"value": nav * 100000}, index=dates)


# ============================================================================
# Trade Analyzer Tests
# ============================================================================
class TestTradeAnalyzer:
    """交易质量分析器测试。"""

    def test_trade_quality_metrics_creation(self):
        """测试 TradeQualityMetrics 创建。"""
        metrics = TradeQualityMetrics(
            trade_id=1,
            entry_date=pd.Timestamp("2023-01-01"),
            exit_date=pd.Timestamp("2023-01-05"),
            entry_price=100.0,
            exit_price=105.0,
            direction=1,
            pnl=500.0,
            pnl_pct=0.05,
            mae=100.0,
            mfe=200.0,
            mae_pct=0.02,
            mfe_pct=0.04,
            efficiency=1.25,
            hold_days=5,
            is_winner=True,
        )

        assert metrics.trade_id == 1
        assert metrics.pnl_pct == 0.05
        assert metrics.is_winner is True
        assert metrics.direction == 1

    def test_trade_quality_summary(self):
        """测试 TradeQualitySummary。"""
        summary = TradeQualitySummary(
            strategy_name="TestStrategy",
            total_trades=100,
            winning_trades=55,
            win_rate=0.55,
            avg_mae=0.03,
            avg_mfe=0.05,
            payoff_ratio=1.5,
        )

        assert summary.strategy_name == "TestStrategy"
        assert summary.win_rate == 0.55
        assert "TestStrategy" in summary.summary()

    def test_analyze_trades(self, sample_trade_log, sample_ohlcv):
        """测试交易分析。"""
        analyzer = TradeAnalyzer()
        metrics = analyzer.analyze_trades(sample_trade_log, sample_ohlcv)

        assert isinstance(metrics, list)
        # 至少能分析出一些交易
        assert len(metrics) > 0

    def test_summarize(self, sample_trade_log, sample_ohlcv):
        """测试交易汇总。"""
        analyzer = TradeAnalyzer()
        metrics = analyzer.analyze_trades(sample_trade_log, sample_ohlcv)
        summary = analyzer.summarize(metrics, strategy_name="TestStrategy")

        assert summary.strategy_name == "TestStrategy"
        assert summary.total_trades == len(metrics)


# ============================================================================
# Market State Analyzer Tests
# ============================================================================
class TestMarketStateAnalyzer:
    """市场状态切割分析器测试。"""

    def test_market_state_enum(self):
        """测试市场状态枚举。"""
        assert MarketState.BULL_QUIET.value == "bull_quiet"
        assert MarketState.CRISIS.value == "crisis"

    def test_market_state_metrics(self):
        """测试市场状态指标。"""
        metrics = MarketStateMetrics(
            state=MarketState.BULL_QUIET,
            state_name="牛市平静",
            days=100,
            days_pct=0.4,
            total_return=0.15,
            sharpe=1.2,
        )

        assert metrics.state == MarketState.BULL_QUIET
        assert metrics.days == 100

    def test_classify_market_states(self, sample_vix_data):
        """测试市场状态分类。"""
        analyzer = MarketStateAnalyzer()
        states = analyzer._classify_market_states(sample_vix_data["vix"])

        assert len(states) == len(sample_vix_data)
        assert all(isinstance(s, MarketState) for s in states)

    def test_analyze_strategy(self, sample_vix_data, sample_equity_curve):
        """测试策略分析。"""
        analyzer = MarketStateAnalyzer()
        analyzer.vix_data = sample_vix_data
        analyzer.state_series = analyzer._classify_market_states(sample_vix_data["vix"])

        summary = analyzer.analyze_strategy(
            equity_curve=sample_equity_curve,
            strategy_name="TestStrategy",
        )

        assert summary.strategy_name == "TestStrategy"
        assert len(summary.state_metrics) > 0


# ============================================================================
# ML Explainer Tests
# ============================================================================
class TestMLExplainer:
    """ML 可解释性分析器测试。"""

    def test_feature_attribution(self):
        """测试特征归因。"""
        attr = FeatureAttribution(
            feature_name="RSI",
            shap_value=0.15,
            contribution_pct=0.25,
            direction="positive",
            importance_rank=1,
        )

        assert attr.feature_name == "RSI"
        assert attr.direction == "positive"

    def test_explain_prediction(self):
        """测试预测解释。"""
        from sklearn.linear_model import LogisticRegression

        # 创建模拟数据
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression()
        model.fit(X, y)

        explainer = MLExplainer()
        result = explainer.explain_prediction(
            model=model,
            X=X[0],
            feature_names=["RSI", "MACD", "Volume", "ATR", "ADX"],
            model_name="LogisticRegression",
        )

        assert result.model_name == "LogisticRegression"
        assert result.prediction_direction in ["buy", "sell", "hold"]
        assert len(result.feature_attributions) > 0

    def test_analyze_feature_importance(self):
        """测试特征重要性分析。"""
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = MLExplainer()
        df = explainer.analyze_feature_importance(
            model=model,
            X=X,
            feature_names=["RSI", "MACD", "Volume", "ATR", "ADX"],
            top_n=5,
        )

        assert len(df) == 5
        assert "feature" in df.columns
        assert "importance" in df.columns


# ============================================================================
# LLM Explainer Tests
# ============================================================================
class TestLLMExplainer:
    """LLM 可解释性分析器测试。"""

    def test_theme_attribution(self):
        """测试主题归因。"""
        attr = ThemeAttribution(
            theme="technical_analysis",
            theme_name_cn="技术分析",
            keyword_count=10,
            keyword_list=["RSI", "MACD"],
            weight=0.4,
            confidence=0.8,
        )

        assert attr.theme == "technical_analysis"
        assert attr.weight == 0.4

    def test_analyze_reasoning(self):
        """测试 reasoning 分析。"""
        reasoning = """
        基于技术分析，RSI 指标显示超卖，MACD 出现金叉。
        美联储鸽派言论降低了加息预期。
        建议：买入
        置信度：75%
        """

        explainer = LLMExplainer()
        result = explainer.analyze_reasoning(
            reasoning_text=reasoning,
            model_name="TestLLM",
        )

        assert result.model_name == "TestLLM"
        assert result.signal in ["buy", "sell", "hold"]
        assert len(result.theme_attributions) > 0

    def test_extract_signal(self):
        """测试信号提取。"""
        explainer = LLMExplainer()

        assert explainer._extract_signal("建议买入") == "buy"
        assert explainer._extract_signal("建议卖出") == "sell"
        assert explainer._extract_signal("持有等待") == "hold"

    def test_extract_confidence(self):
        """测试置信度提取。"""
        explainer = LLMExplainer()

        conf = explainer._extract_confidence("置信度：75%")
        assert conf == 0.75

        conf = explainer._extract_confidence("confidence: 80%")
        assert conf == 0.80


# ============================================================================
# Cross Market Analyzer Tests
# ============================================================================
class TestCrossMarketAnalyzer:
    """跨市场分析器测试。"""

    def test_signal_decay_result(self):
        """测试信号衰减结果。"""
        result = SignalDecayResult(
            strategy="LightGBM",
            market="A_share",
            signal_date=pd.Timestamp("2023-01-01"),
            signal_strength=0.8,
            return_t=0.01,
            return_t1=0.005,
            return_t3=0.02,
            return_t5=0.03,
        )

        assert result.strategy == "LightGBM"
        assert result.market == "A_share"

    def test_cross_market_summary(self):
        """测试跨市场汇总。"""
        summary = CrossMarketSummary(
            strategy="LightGBM",
            a_share_sharpe=1.2,
            us_market_sharpe=1.0,
            zero_shot_score=0.83,
        )

        assert summary.strategy == "LightGBM"
        assert summary.zero_shot_score == 0.83

    def test_compare_markets(self):
        """测试市场对比。"""
        analyzer = CrossMarketAnalyzer()
        summary = analyzer.compare_markets(
            a_share_sharpe=1.2,
            a_share_return=0.25,
            us_market_sharpe=1.0,
            us_market_return=0.20,
            strategy_name="LightGBM",
        )

        assert summary.strategy == "LightGBM"
        assert summary.sharpe_gap == pytest.approx(0.2, rel=1e-5)
        assert summary.zero_shot_score > 0


# ============================================================================
# Attribution Comparator Tests
# ============================================================================
class TestAttributionComparator:
    """归因对比分析器测试。"""

    def test_attribution_alignment(self):
        """测试归因对齐。"""
        alignment = AttributionAlignment(
            feature="RSI",
            ml_contribution=0.15,
            llm_contribution=0.20,
            alignment_score=0.85,
            alignment_type="aligned",
        )

        assert alignment.feature == "RSI"
        assert alignment.alignment_type == "aligned"

    def test_compare(self):
        """测试归因对比。"""
        ml_result = MLExplanationResult(
            model_name="LightGBM",
            prediction_direction="buy",
            confidence=0.75,
            feature_attributions=[
                FeatureAttribution("RSI", 0.15, 0.25, "positive", 1),
            ],
            top_features=["RSI"],
        )

        llm_result = LLMExplanationResult(
            model_name="DeepSeek",
            signal="buy",
            confidence=0.80,
            reasoning="RSI 显示买入信号",
            theme_attributions=[
                ThemeAttribution("technical_analysis", "技术分析", 5, ["RSI"], 0.5, 0.7),
            ],
            key_factors=["RSI"],
        )

        comparator = AttributionComparator()
        result = comparator.compare(ml_result, llm_result)

        assert result.decision_consistency is True
        assert len(result.alignments) > 0

    def test_check_consistency(self):
        """测试一致性检查。"""
        comparator = AttributionComparator()

        assert comparator._check_consistency("buy", "buy") is True
        assert comparator._check_consistency("sell", "sell") is True
        assert comparator._check_consistency("buy", "sell") is False


# ============================================================================
# Advanced Visualizer Tests
# ============================================================================
class TestAdvancedVisualizer:
    """高级可视化器测试。"""

    def test_visualization_config(self):
        """测试可视化配置。"""
        config = VisualizationConfig(
            output_dir="test_output",
            dpi=150,
        )

        assert config.output_dir == "test_output"
        assert config.dpi == 150

    def test_plot_trade_quality_comparison(self, tmp_path):
        """测试交易质量对比图。"""
        config = VisualizationConfig(output_dir=str(tmp_path))
        visualizer = AdvancedVisualizer(config=config)

        summaries = {
            "StrategyA": TradeQualitySummary(
                strategy_name="StrategyA",
                win_rate=0.55,
                payoff_ratio=1.5,
                avg_mae=0.03,
                avg_mfe=0.05,
                avg_efficiency=0.6,
            ),
        }

        path = visualizer.plot_trade_quality_comparison(summaries)
        assert path.exists()


# ============================================================================
# Integration Tests
# ============================================================================
class TestIntegration:
    """集成测试。"""

    def test_full_evaluation_pipeline(self, sample_ohlcv, sample_trade_log, sample_vix_data, sample_equity_curve):
        """测试完整评估流程。"""
        # 1. 交易质量分析
        trade_analyzer = TradeAnalyzer()
        trade_metrics = trade_analyzer.analyze_trades(sample_trade_log, sample_ohlcv)
        trade_summary = trade_analyzer.summarize(trade_metrics, "TestStrategy")

        assert trade_summary.total_trades > 0

        # 2. 市场状态分析
        market_analyzer = MarketStateAnalyzer()
        market_analyzer.vix_data = sample_vix_data
        market_analyzer.state_series = market_analyzer._classify_market_states(sample_vix_data["vix"])
        market_summary = market_analyzer.analyze_strategy(
            equity_curve=sample_equity_curve,
            strategy_name="TestStrategy",
        )

        assert market_summary.strategy_name == "TestStrategy"

        # 3. LLM 分析
        llm_explainer = LLMExplainer()
        llm_result = llm_explainer.analyze_reasoning(
            "RSI 超卖，建议买入。置信度 70%",
            "TestLLM",
        )

        assert llm_result.signal in ["buy", "sell", "hold"]

        # 4. 跨市场分析
        cross_analyzer = CrossMarketAnalyzer()
        cross_summary = cross_analyzer.compare_markets(
            a_share_sharpe=1.0,
            a_share_return=0.1,
            us_market_sharpe=0.8,
            us_market_return=0.08,
            strategy_name="TestStrategy",
        )

        assert cross_summary.zero_shot_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])