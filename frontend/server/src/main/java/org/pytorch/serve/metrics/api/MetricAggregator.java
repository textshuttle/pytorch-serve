package org.pytorch.serve.metrics.api;

import org.pytorch.serve.metrics.format.prometheous.PrometheusMetricManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Priority;

public final class MetricAggregator {

    private MetricAggregator() {}

    // Is executed upon successful Job insertion in queue
    public static void handleInferenceMetric(final String modelName, final String modelVersion, Priority priority) {
        ConfigManager configMgr = ConfigManager.getInstance();
        if (configMgr.isMetricApiEnable()
                && configMgr.getMetricsFormat().equals(ConfigManager.METRIC_FORMAT_PROMETHEUS)) {
            PrometheusMetricManager metrics = PrometheusMetricManager.getInstance();
            metrics.incInferCount(modelName, modelVersion);
            metrics.incQueueCount(modelName, modelVersion, priority);
        }
    }

    // Is executed upon successful Job completion
    public static void handleInferenceMetric(
            final String modelName, final String modelVersion, Priority priority, long timeInQueue, long inferTime) {
        ConfigManager configMgr = ConfigManager.getInstance();
        if (configMgr.isMetricApiEnable()
                && configMgr.getMetricsFormat().equals(ConfigManager.METRIC_FORMAT_PROMETHEUS)) {
            PrometheusMetricManager metrics = PrometheusMetricManager.getInstance();
            metrics.incInferLatency(inferTime, modelName, modelVersion);
            metrics.incQueueLatency(timeInQueue, modelName, modelVersion, priority);
            metrics.decQueueCount(modelName, modelVersion, priority);
        }
    }
}
