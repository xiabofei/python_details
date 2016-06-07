/*
 * 目的是理清代理模式之延迟加载技术
 * https://www.ibm.com/developerworks/cn/java/j-lo-proxy-pattern/
 * */

// 定义一个DB查询接口
public interface IDBQuery{
    String request();
}

public class DBQuery implements IDBQuery{
    public DBQuery(){
        try{
            sleep(1000)
        }catch()
        {}
    }
    @Override
    public String request()
    {}
}

public class DBQueryProxy implements IDBQuery{
    // 判断是否创建过真实DBQuery的实例
    private DBQuery real = null;
    // 在代理的接口中完成两件事情
    // 1. 产生一个真实对象
    // 2. 调用真实对象的方法
    @Override
    public String requestt()
    {
        // 1. 耗时操作
        if real==null:
            real = new DBQuery(); 
        // 2. 调用真实对象的方法
        return real.request();
    }
}

public class Main{
    public static void main(String[] args){
        IDBQuery q = new DBQueryProxy();
        q.requestt()
    }
}
