package org.pytorch.serve.util;

import java.util.Map;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
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
    private int minFreeMemory;
    private float maxShareFailures;
    private AtomicInteger[] nFailures;
    private AtomicInteger[] freeMemory;
    private ConcurrentHashMap<String, Integer> workerIds;

    private GPUManager(int nGPUs, int minFreeMemory, float maxShareFailures) {
        this.nGPUs = nGPUs;
        this.minFreeMemory = minFreeMemory;
        this.maxShareFailures = maxShareFailures;

        if (nGPUs > 0) {
            this.workerIds = new ConcurrentHashMap<String, Integer> ();
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

    public static void init(ConfigManager configManager) {
        int nGPUs = configManager.getNumberOfGpu();
        int minFreeMemory = configManager.getMinFreeGpuMemory();
        float maxShareFailures = configManager.getMaxShareGpuFailures();
        instance = new GPUManager(nGPUs, minFreeMemory, maxShareFailures);
    }

    public static GPUManager getInstance() {
        return instance;
    }

    public int getGPU(String workerId) {
        int gpuId = -1;
        // return -1 if there are no gpus
        if (this.nGPUs == 0) {
            return gpuId;
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
        // get total failures for share calculation
        int nFailuresSum = 0;
        for (int i = 0; i < this.nGPUs; i++) {
            nFailuresSum += this.nFailures[i].intValue();
        }
        // get free memory for all eligible GPUs
        HashMap<Integer, Integer> eligibleIdFreeMems = new HashMap<Integer, Integer> ();
        for (int i = 0; i < this.nGPUs; i++) {
            // check that free memory is available and exceeds minimum
            if (this.freeMemory[i].intValue() > this.minFreeMemory) {
                // check that share of failures is smaller than maximum
                float shareFailures = (float) this.nFailures[i].intValue() / (float) nFailuresSum;
                if (shareFailures < this.maxShareFailures) {
                    eligibleIdFreeMems.put(i, this.freeMemory[i].intValue());
                } else {
                    logger.warn("GPU ID " + String.valueOf(i) + " deemed ineligible since it accounts for at least " + 
                        String.valueOf(this.maxShareFailures) + " of failures");
                }
            }
        }
        // fork on number of eligible GPUs
        if (eligibleIdFreeMems.size() == 0) {
            logger.error("No eligible GPUs available");
        } else if (eligibleIdFreeMems.size() == 1) {
            gpuId = eligibleIdFreeMems.keySet().iterator().next();
        } else {
            // get sum of eligible id free memory for prob calculation
            int eligibleIdFreeMemSum = 0;
            for (Map.Entry<Integer, Integer> entry : eligibleIdFreeMems.entrySet()) {
                eligibleIdFreeMemSum += entry.getValue();
            }
            // store cumulative probabilities in navigable map
            float cumProb = 0.0f;
            TreeMap<Float, Integer> cumProbIds = new TreeMap<Float, Integer> ();
            for (Map.Entry<Integer, Integer> entry : eligibleIdFreeMems.entrySet()) {
                int i = entry.getKey();
                int freeMem = entry.getValue();
                cumProbIds.put(cumProb, i);
                cumProb += (float) freeMem / (float) eligibleIdFreeMemSum;
            }
            // make random selection
            float randFloat = ThreadLocalRandom.current().nextFloat();
            gpuId = cumProbIds.ceilingEntry(randFloat).getValue();
            logger.info("Assigning gpuId " + String.valueOf(gpuId) + 
                        " with free memory " + String.valueOf(eligibleIdFreeMems.get(gpuId)) + 
                        " with number of failures " + String.valueOf(this.nFailures[gpuId].intValue()) + 
                        " to workerId " + workerId);
        }
        return gpuId;
    }
}
