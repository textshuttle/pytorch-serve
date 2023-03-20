package org.pytorch.serve.metrics.format.prometheous;

import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Priority;
import io.prometheus.client.Counter;
import io.prometheus.client.Gauge;
import java.util.HashMap;
import java.util.UUID;

public final class PrometheusMetricManager {

    private static final PrometheusMetricManager METRIC_MANAGER = new PrometheusMetricManager();
    private static final String METRICS_UUID = UUID.randomUUID().toString();
    private final Counter inferRequestCount;
    private final Counter inferLatency;
    private final Counter queueLatency;
    private final HashMap<Priority, Gauge> queueRequestCounts;

    private PrometheusMetricManager() {
        String[] metricsLabelsNonQueue = {"uuid", "model_name", "model_version"};
        inferRequestCount =
                Counter.build()
                        .name("ts_inference_requests_total")
                        .labelNames(metricsLabelsNonQueue)
                        .help("Total number of inference requests.")
                        .register();
        inferLatency =
                Counter.build()
                        .name("ts_inference_latency_microseconds")
                        .labelNames(metricsLabelsNonQueue)
                        .help("Cumulative inference duration in microseconds.")
                        .register();

        String[] metricsLabelsQueue = {"uuid", "model_name", "model_version", "priority"};
        queueLatency =
                Counter.build()
                        .name("ts_queue_latency_microseconds")
                        .labelNames(metricsLabelsQueue)
                        .help("Cumulative queue duration in microseconds.")
                        .register();
        queueRequestCounts = new HashMap<Priority, Gauge> ();
        for (Priority priority : Priority.values()) { 
                queueRequestCounts.put(priority, 
                        Gauge.build()
                                .name("ts_queue_requests_" + priority.toString().toLowerCase() + "_total")
                                .labelNames(metricsLabelsQueue)
                                .help("Current queue inference request count.")
                                .register());
        }
    }

    private static String getOrDefaultModelVersion(String modelVersion) {
        return modelVersion == null ? "default" : modelVersion;
    }

    public static PrometheusMetricManager getInstance() {
        return METRIC_MANAGER;
    }

    /**
     * Counts the time in ns it took for an inference to be completed
     *
     * @param inferTime time in nanoseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incInferLatency(long inferTime, String modelName, String modelVersion) {
        inferLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(inferTime / 1000.0);
    }

    /**
     * Counts the time in ns an inference request was queued before being executed
     *
     * @param queueTime time in nanoseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incQueueLatency(long queueTime, String modelName, String modelVersion) {
        int queueSize = ConfigManager.getInstance().getJobQueueSize();
        queueLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion), String.valueOf(queueSize))
                .inc(queueTime / 1000.0);
    }

    /**
     * Counts a valid inference request to be processed
     *
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incInferCount(String modelName, String modelVersion) {
        inferRequestCount
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc();
    }


    /**
     * Counts a valid inference request that has been added to a queue
     *
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incQueueCount(String modelName, String modelVersion, Priority priority) {
        int queueSize = ConfigManager.getInstance().getJobQueueSize();
        queueRequestCounts.get(priority)
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion), String.valueOf(queueSize))
                .inc();
    }

    /**
     * Counts a valid inference request that has been removed from a queue
     *
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void decQueueCount(String modelName, String modelVersion, Priority priority) {
        int queueSize = ConfigManager.getInstance().getJobQueueSize();
        queueRequestCounts.get(priority)
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion), String.valueOf(queueSize))
                .dec();
    }
}
