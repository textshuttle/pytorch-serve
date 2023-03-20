package org.pytorch.serve.job;

import java.util.Map;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.util.Prioritisable;
import org.pytorch.serve.util.Priority;

public abstract class Job implements Prioritisable {

    private String modelName;
    private String modelVersion;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestInput input;
    private Priority priority;
    private long begin;
    private long scheduled;

    public Job(String modelName, String version, WorkerCommands cmd, RequestInput input) {
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;
        this.modelVersion = version;
        begin = System.nanoTime();
        scheduled = begin;
        this.priority = Priority.valueOf(input.getHeaders().getOrDefault("x-ts-priority", "MAX").toUpperCase());
    }

    public Priority getPriority() {
        return this.priority;
    }

    public void setPriority(Priority priority) {
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
