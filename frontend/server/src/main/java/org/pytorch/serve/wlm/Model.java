package org.pytorch.serve.wlm;

import com.google.common.annotations.VisibleForTesting;
import com.google.gson.JsonObject;
import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.PriorityLinkedBlockingDeque;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Model {

    public static final String DEFAULT_DATA_QUEUE = "DATA_QUEUE";

    public static final String MIN_WORKERS = "minWorkers";
    public static final String MAX_WORKERS = "maxWorkers";
    public static final String BATCH_SIZE = "batchSize";
    public static final String MAX_BATCH_DELAY = "maxBatchDelay";
    public static final String RESPONSE_TIMEOUT = "responseTimeout";
    public static final String QUEUE_TIMEOUT = "queueTimeout";
    public static final String DEFAULT_VERSION = "defaultVersion";
    public static final String MAR_NAME = "marName";

    private static final Logger logger = LoggerFactory.getLogger(Model.class);

    private ModelArchive modelArchive;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private ReentrantLock lock;
    private int responseTimeout;
    private int queueTimeout;
    private int queueSize;
    private float highPrioProb;
    private ModelVersionName modelVersionName;

    private boolean isWorkflowModel;

    // Total number of subsequent inference request failures
    private AtomicInteger failedInfReqs;

    // Per worker thread job queue. This separates out the control queue from data queue
    private ConcurrentMap<String, PriorityLinkedBlockingDeque<Job>> jobsDb;

    public Model(ModelArchive modelArchive, int queueSize, float highPrioProb) {
        this.modelArchive = modelArchive;
        this.queueSize = queueSize;
        this.highPrioProb = highPrioProb;
        batchSize = 1;
        maxBatchDelay = 100;
        jobsDb = new ConcurrentHashMap<>();
        // Always have a queue for data
        jobsDb.putIfAbsent(DEFAULT_DATA_QUEUE, new PriorityLinkedBlockingDeque<>(this.queueSize, this.highPrioProb));
        failedInfReqs = new AtomicInteger(0);
        lock = new ReentrantLock();
        modelVersionName =
                new ModelVersionName(
                        this.modelArchive.getModelName(), this.modelArchive.getModelVersion());
    }

    public JsonObject getModelState(boolean isDefaultVersion) {

        JsonObject modelInfo = new JsonObject();
        modelInfo.addProperty(DEFAULT_VERSION, isDefaultVersion);
        modelInfo.addProperty(MAR_NAME, FilenameUtils.getName(getModelUrl()));
        modelInfo.addProperty(MIN_WORKERS, getMinWorkers());
        modelInfo.addProperty(MAX_WORKERS, getMaxWorkers());
        modelInfo.addProperty(BATCH_SIZE, getBatchSize());
        modelInfo.addProperty(MAX_BATCH_DELAY, getMaxBatchDelay());
        modelInfo.addProperty(RESPONSE_TIMEOUT, getResponseTimeout());
        modelInfo.addProperty(QUEUE_TIMEOUT, getQueueTimeout());

        return modelInfo;
    }

    public void setModelState(JsonObject modelInfo) {
        minWorkers = modelInfo.get(MIN_WORKERS).getAsInt();
        maxWorkers = modelInfo.get(MAX_WORKERS).getAsInt();
        maxBatchDelay = modelInfo.get(MAX_BATCH_DELAY).getAsInt();
        responseTimeout = modelInfo.get(RESPONSE_TIMEOUT).getAsInt();
        queueTimeout = modelInfo.get(QUEUE_TIMEOUT).getAsInt();
        batchSize = modelInfo.get(BATCH_SIZE).getAsInt();
    }

    public String getModelName() {
        return modelArchive.getModelName();
    }

    public ModelVersionName getModelVersionName() {
        return modelVersionName;
    }

    public String getVersion() {
        return modelArchive.getModelVersion();
    }

    public File getModelDir() {
        return modelArchive.getModelDir();
    }

    public String getModelUrl() {
        return modelArchive.getUrl();
    }

    public ModelArchive getModelArchive() {
        return modelArchive;
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    public boolean isWorkflowModel() {
        return isWorkflowModel;
    }

    public void setWorkflowModel(boolean workflowModel) {
        isWorkflowModel = workflowModel;
    }

    public void addJob(String threadId, Job job) {
        PriorityLinkedBlockingDeque<Job> blockingDeque = jobsDb.get(threadId);
        if (blockingDeque == null) {
            blockingDeque = new PriorityLinkedBlockingDeque<>(this.queueSize, this.highPrioProb);
            jobsDb.put(threadId, blockingDeque);
        }
        blockingDeque.offer(job);
    }

    public void removeJobQueue(String threadId) {
        if (!threadId.equals(DEFAULT_DATA_QUEUE)) {
            jobsDb.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return jobsDb.get(DEFAULT_DATA_QUEUE).offer(job);
    }

    public void addFirst(Job job) {
        jobsDb.get(DEFAULT_DATA_QUEUE).addFirst(job);
    }

    public void pollBatch(String threadId, long waitTime, Map<String, Job> jobsRepo)
            throws InterruptedException {
        if (jobsRepo == null || threadId == null || threadId.isEmpty()) {
            throw new IllegalArgumentException("Invalid input given provided");
        }

        if (!jobsRepo.isEmpty()) {
            throw new IllegalArgumentException(
                    "The jobs repo provided contains stale jobs. Clear them!!");
        }

        PriorityLinkedBlockingDeque<Job> jobsQueue = jobsDb.get(threadId);
        if (jobsQueue != null && !jobsQueue.isEmpty()) {
            Job j = jobsQueue.poll(waitTime, TimeUnit.MILLISECONDS);
            if (j != null) {
                jobsRepo.put(j.getJobId(), j);
                return;
            }
        }

        try {
            lock.lockInterruptibly();
            long maxDelay = maxBatchDelay;
            jobsQueue = jobsDb.get(DEFAULT_DATA_QUEUE);

            long requestTimeoutNs = queueTimeout * (long) 1E6;
            Job j = jobsQueue.poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            if (j != null) {
                if (addToBatchIfNotStale(j, jobsRepo, System.nanoTime(), requestTimeoutNs)) {
                    logger.trace("get first job: {}", j.getJobId());
                }
            }

            // describe request job batch size always is 1
            if (j.getCmd() == WorkerCommands.DESCRIBE) {
                return;
            }
            long begin = System.currentTimeMillis();
            for (int i = 0; i < batchSize - 1; ++i) {
                j = jobsQueue.poll(maxDelay, TimeUnit.MILLISECONDS);
                if (j == null) {
                    break;
                }
                long end = System.currentTimeMillis();
                // describe request job batch size always is 1
                if (j.getCmd() == WorkerCommands.DESCRIBE) {
                    // Add the job back into the jobsQueue
                    jobsQueue.addFirst(j);
                    break;
                }
                maxDelay -= end - begin;
                begin = end;
                addToBatchIfNotStale(j, jobsRepo, System.nanoTime(), requestTimeoutNs);
                if (maxDelay <= 0) {
                    break;
                }
            }
            logger.trace("sending jobs, size: {}", jobsRepo.size());
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    @VisibleForTesting
    public static boolean addToBatchIfNotStale(
            Job job, Map<String, Job> jobsRepo, long nowInNs, long timeoutInNs) {
        if ((nowInNs - job.getBegin()) >= timeoutInNs) {
            logger.trace("Discarding job {} because it is stale", job.getJobId());
            return false;
        }
        jobsRepo.put(job.getJobId(), job);
        return true;
    }

    public int incrFailedInfReqs() {
        return failedInfReqs.incrementAndGet();
    }

    public void resetFailedInfReqs() {
        failedInfReqs.set(0);
    }

    public int getResponseTimeout() {
        return ConfigManager.getInstance().isDebug() ? Integer.MAX_VALUE : responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }

    public int getQueueTimeout() {
        return ConfigManager.getInstance().isDebug() ? Integer.MAX_VALUE : queueTimeout;
    }

    public void setQueueTimeout(int queueTimeout) {
        this.queueTimeout = queueTimeout;
    }
}
