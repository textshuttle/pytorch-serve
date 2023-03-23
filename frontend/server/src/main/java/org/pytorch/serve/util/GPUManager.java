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
    private static final int nFailureHistory = 100;

    private static GPUManager instance;

    private final int nGPUs;
    private final int minFreeMemory;
    private final float maxShareFailures;

    private AtomicInteger[] freeMemory;
    private HashMap<String, Integer> workerIds;
    private ArrayDeque<Integer> gpuFailureHistory;

    private GPUManager(int nGPUs, int minFreeMemory, float maxShareFailures) {
        this.nGPUs = nGPUs;
        this.minFreeMemory = minFreeMemory;
        this.maxShareFailures = maxShareFailures;

        this.gpuFailureHistory = new ArrayDeque<> ();
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
        float maxShareFailures = configManager.getMaxShareGpuFailures();
        instance = new GPUManager(nGPUs, minFreeMemory, maxShareFailures);
    }

    public static synchronized GPUManager getInstance() {
        return instance;
    }

    public synchronized int getGPU(String workerId) {
        // return -1 if there are no gpus
        if (this.nGPUs == 0) {
            return -1;
        }
        int failedGpuId;
        // if the worker was previously assigned to a GPU and now requests a new one, it has likely failed
        // add failed gpu id to failure history, removing old entries to make space if necessary
        if (this.workerIds.containsKey(workerId)) {
            failedGpuId = this.workerIds.get(workerId);
            while (this.gpuFailureHistory.size() > nFailureHistory - 1) {
                this.gpuFailureHistory.removeFirst();
            }
            this.gpuFailureHistory.addLast(failedGpuId);
        }
        // get free memory per GPU
        for (int i = 0; i < this.nGPUs; i++) {
            this.freeMemory[i].set(queryNvidiaSmiFreeMemory(i));
        }
        // get failures for share calculation
        int[] nFailures = new int[this.nGPUs];
        for (Iterator<Integer> iter = this.gpuFailureHistory.iterator(); iter.hasNext();) {
            failedGpuId = iter.next();
            nFailures[failedGpuId]++;
        }
        // get free memory for all eligible GPUs
        HashMap<Integer, Integer> eligibleIdFreeMems = new HashMap<Integer, Integer> ();
        for (int i = 0; i < this.nGPUs; i++) {
            // check that free memory is available and exceeds minimum
            if (this.freeMemory[i].intValue() > this.minFreeMemory) {
                if (this.gpuFailureHistory.size() > 1) {
                    // check that share of failures is smaller than maximum
                    float shareFailures = (float) nFailures[i] / (float) this.gpuFailureHistory.size();
                    if (shareFailures < this.maxShareFailures) {
                        eligibleIdFreeMems.put(i, this.freeMemory[i].intValue());
                    } else {
                        logger.warn("GPU ID {} deemed ineligible since it accounts for {} out of failures {}", i, nFailures[i], this.maxShareFailures);
                    }
                } else {
                    eligibleIdFreeMems.put(i, this.freeMemory[i].intValue());
                }
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
                cumProbIds.put(cumProb, i);
                logger.info("cumProbIds[{}] {} because of freeMem {}", cumProb, i, freeMem);
                cumProb += (float) freeMem / (float) eligibleIdFreeMemSum;
            }
            // make random selection
            float randFloat = ThreadLocalRandom.current().nextFloat();
            logger.info("randFloat {}", randFloat);
            gpuId = cumProbIds.ceilingEntry(randFloat).getValue();
            logger.info("gpuId {}", gpuId);
        }
        logger.info("Assigning gpuId " + gpuId + 
                    " with free memory " + eligibleIdFreeMems.get(gpuId) + 
                    " with number of failures " + nFailures[gpuId] + 
                    " to workerId " + workerId);
        return gpuId;
    }
}
