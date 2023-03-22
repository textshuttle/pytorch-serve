package org.pytorch.serve.util;

public interface Prioritisable {

    public Priority getPriority();
    public void setPriority(Priority priority);

}
