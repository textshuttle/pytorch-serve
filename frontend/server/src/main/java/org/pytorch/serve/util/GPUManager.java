package org.pytorch.serve.util;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class GPUManager {

    private static final Logger logger = LoggerFactory.getLogger(GPUManager.class);

    private static GPUManager instance;

    private int nGPUs;
    private int[] memoryUsage;
    private int[] nFailures;
    private AtomicInteger gpuCounter;

    private GPUManager(int nGPUs) {
        this.nGPUs = nGPUs;
        if (nGPUs > 0) {
            this.gpuCounter = new AtomicInteger(0);
            this.memoryUsage = new int[this.nGPUs];
            this.nFailures = new int[this.nGPUs];
        }
    }

    public static void init(int nGPUs) {
        instance = new GPUManager(nGPUs);
    }

    public static GPUManager getInstance() {
        return instance;
    }

    public int getGPU(String workerId) {
        int gpuId = -1;
        if (this.nGPUs > 0) {
            gpuId = gpuCounter.accumulateAndGet(this.nGPUs, (prev, maxGpuId) -> ++prev % maxGpuId);
        }
        logger.info("Assigning gpuId " + String.valueOf(gpuId) + " to workerId " + workerId);
        return gpuId;
    }

}
