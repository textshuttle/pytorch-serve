package org.pytorch.serve.util;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ConcurrentHashMap;
import java.nio.charset.StandardCharsets;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class GPUManager {

    private static final Logger logger = LoggerFactory.getLogger(GPUManager.class);

    private static GPUManager instance;

    private int nGPUs;
    private AtomicInteger[] nFailures;
    private AtomicInteger[] freeMemory;
    private ConcurrentHashMap<String, Integer> workerIds;

    private GPUManager(int nGPUs) {
        this.nGPUs = nGPUs;
        this.workerIds = new ConcurrentHashMap<String, Integer>();
        if (nGPUs > 0) {
            this.nFailures = new AtomicInteger[this.nGPUs];
            this.freeMemory = new AtomicInteger[this.nGPUs];
            for (int i = 0; i < this.nGPUs; i++) {
                this.nFailures[i] = new AtomicInteger(0);
                this.freeMemory[i] = new AtomicInteger(-1);
            }
        }
    }

    // code largely copied from WorkerThread::getGpuUsage
    private int queryNvidiaSmiFreeMemory(int gpuId) {
        Process process;
        try {
            process =
                    Runtime.getRuntime()
                            .exec(
                                    "nvidia-smi -i "
                                            + gpuId
                                            + " --query-gpu=memory.free --format=csv,noheader,nounits");
            process.waitFor();
            int exitCode = process.exitValue();
            if (exitCode != 0) {
                InputStream error = process.getErrorStream();
                for (int i = 0; i < error.available(); i++) {
                    logger.error("" + error.read());
                }
                return -1;
            }
            InputStream stdout = process.getInputStream();
            BufferedReader reader =
                    new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line = reader.readLine();
            if (line == null) {
                return -1;
            } else {
                return Integer.parseInt(line);
            }
        } catch (Exception e) {
            logger.error("Exception raised : " + e.toString());
        }

        return -1;
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
        // get free memory per GPU
        for (int i = 0; i < this.nGPUs; i++) {
            this.freeMemory[i].set(queryNvidiaSmiFreeMemory(i));
        }
        // assign GPU id (dummy for now)
        int gpuId = 0;
        logger.info("Assigning gpuId " + String.valueOf(gpuId) + " to workerId " + workerId);
        return gpuId;
    }
}
