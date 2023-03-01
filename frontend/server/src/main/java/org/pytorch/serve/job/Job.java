package org.pytorch.serve.job;

import java.util.Map;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;

public abstract class Job {

    private String modelName;
    private String modelVersion;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestInput input;
    private int priority;
    private long begin;
    private long scheduled;

    public Job(String modelName, String version, WorkerCommands cmd, RequestInput input) {
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;
        this.modelVersion = version;
        begin = System.nanoTime();
        scheduled = begin;

        Map<String, String> headers = input.getHeaders();
        if (headers.containsKey("X-Priority")) {
            this.priority = Integer.parseInt(headers.get("X-Priority"));
        } else {
            this.priority = 0;
        }
    }

    public int getPriority() {
        return this.priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public String getJobId() {
        return input.getRequestId();
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public WorkerCommands getCmd() {
        return cmd;
    }

    public boolean isControlCmd() {
        return !WorkerCommands.PREDICT.equals(cmd) && !WorkerCommands.DESCRIBE.equals(cmd);
    }

    public RequestInput getPayload() {
        return input;
    }

    public void setScheduled() {
        scheduled = System.nanoTime();
    }

    public long getBegin() {
        return begin;
    }

    public long getScheduled() {
        return scheduled;
    }

    public abstract void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders);

    public abstract void sendError(int status, String error);
}
