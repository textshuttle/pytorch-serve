package org.pytorch.serve.util;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class GPUManager {

    private static final Logger logger = LoggerFactory.getLogger(GPUManager.class);

    private static GPUManager instance;

    private int nGPUs;
    private AtomicInteger[] nFailures;
    private AtomicInteger[] memoryUsage;
    private ConcurrentHashMap<String, Integer> workerIds;

    private GPUManager(int nGPUs) {
        this.nGPUs = nGPUs;
        this.workerIds = new ConcurrentHashMap<String, Integer>();
        if (nGPUs > 0) {
            this.nFailures = new AtomicInteger[this.nGPUs];
            this.memoryUsage = new AtomicInteger[this.nGPUs];
            for (int i = 0; i < this.nGPUs; i++) {
                this.nFailures[i] = new AtomicInteger(0);
                this.memoryUsage[i] = new AtomicInteger(-1);
            }
        }
    }

    private int queryNvidiaSmiMemory(int gpuId) {
        // dummy for now
        // returns -1 if memory unavailable, else memory in bytes
        return 100;
    }

    public static void init(int nGPUs) {
        instance = new GPUManager(nGPUs);
    }

    public static GPUManager getInstance() {
        return instance;
    }

    public int getGPU(String workerId) {
        // return -1 if there are no gpus
        if (this.nGPUs == 0) {
            return -1;
        }
        // if the worker was previously assigned to a GPU and now requests a new one, it has likely failed
        // increment failure counter for the previously assigned GPU
        if (this.workerIds.containsKey(workerId)) {
            this.nFailures[this.workerIds.get(workerId)].incrementAndGet();
        }
        // get GPU memory usage per GPU
        for (int i = 0; i < this.nGPUs; i++) {
            this.memoryUsage[i].set(queryNvidiaSmiMemory(i));
        }
        // assign GPU id (dummy for now)
        int gpuId = 0;
        logger.info("Assigning gpuId " + String.valueOf(gpuId) + " to workerId " + workerId);
        return gpuId;
    }
}
