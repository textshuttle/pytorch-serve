package org.pytorch.serve.util;

import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Iterator;
import java.util.ArrayDeque;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.nio.charset.StandardCharsets;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class GPUManager {

    private static final Logger logger = LoggerFactory.getLogger(GPUManager.class);

    private static GPUManager instance;

    private final int nGPUs;
    private final int minFreeMemory;
    private final int overrideGpuId;

    private AtomicInteger[] freeMemory;
    private HashMap<String, Integer> workerIds;

    private GPUManager(int nGPUs, int minFreeMemory, int overrideGpuId) {
        this.nGPUs = nGPUs;
        this.minFreeMemory = minFreeMemory;
        this.overrideGpuId = overrideGpuId;

        this.workerIds = new HashMap<> ();

        if (nGPUs > 0) {
            this.freeMemory = new AtomicInteger[this.nGPUs];
            for (int i = 0; i < this.nGPUs; i++) {
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
            logger.error("An exception occurred when querying for free gpu memory", e);
        }

        return -1;
    }

    public static synchronized void init(ConfigManager configManager) {
        int nGPUs = configManager.getNumberOfGpu();
        int minFreeMemory = configManager.getMinFreeGpuMemory();
        int override_gpu_id = configManager.getOverrideGpuId();
        instance = new GPUManager(nGPUs, minFreeMemory, override_gpu_id);
    }

    public static synchronized GPUManager getInstance() {
        return instance;
    }

    public synchronized int getGPU(String workerId) {
        // return -1 if there are no gpus
        if (this.nGPUs == 0) {
            logger.error("No eligible GPUs available, falling back to CPU");
            return -1;
        }
        // return override if given
        if (this.overrideGpuId > -1) {
            return this.overrideGpuId;
        }
        // get free memory per GPU
        for (int i = 0; i < this.nGPUs; i++) {
            this.freeMemory[i].set(queryNvidiaSmiFreeMemory(i));
        }

        // get free memory for all eligible GPUs
        HashMap<Integer, Integer> eligibleIdFreeMems = new HashMap<Integer, Integer> ();
        for (int i = 0; i < this.nGPUs; i++) {
            // check that free memory is available and exceeds minimum
            if (this.freeMemory[i].intValue() > this.minFreeMemory) {
                eligibleIdFreeMems.put(i, this.freeMemory[i].intValue());
                logger.info("eligibleIdFreeMems[{}] {}", i, this.freeMemory[i].intValue());

            }
        }
        logger.info("eligibleIdFreeMems.size() {}", eligibleIdFreeMems.size());
        // fork on number of eligible GPUs
        int gpuId = -1;
        if (eligibleIdFreeMems.size() == 0) {
            logger.error("No eligible GPUs available, falling back to CPU");
            return gpuId;
        }
        if (eligibleIdFreeMems.size() == 1) {
            gpuId = eligibleIdFreeMems.keySet().iterator().next();
        } else {
            // get sum of eligible id free memory for prob calculation
            int eligibleIdFreeMemSum = 0;
            for (Map.Entry<Integer, Integer> entry : eligibleIdFreeMems.entrySet()) {
                eligibleIdFreeMemSum += entry.getValue();
            }
            logger.info("eligibleIdFreeMemSum {}", eligibleIdFreeMemSum);
            // store cumulative probabilities in navigable map
            float cumProb = 0.0f;
            TreeMap<Float, Integer> cumProbIds = new TreeMap<Float, Integer> ();
            for (Map.Entry<Integer, Integer> entry : eligibleIdFreeMems.entrySet()) {
                int i = entry.getKey();
                int freeMem = entry.getValue();
                cumProb += (float) freeMem / (float) eligibleIdFreeMemSum;
                cumProbIds.put(cumProb, i);
            }
            // make random selection
            float randFloat = ThreadLocalRandom.current().nextFloat();
            logger.info("randFloat {}", randFloat);
            gpuId = cumProbIds.ceilingEntry(randFloat).getValue();
            logger.info("gpuId {}", gpuId);
        }
        logger.info("Assigning gpuId " + gpuId + 
                    " with free memory " + eligibleIdFreeMems.get(gpuId) + 
                    " to workerId " + workerId);
        return gpuId;
    }
}
