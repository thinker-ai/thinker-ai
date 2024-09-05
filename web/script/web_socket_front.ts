import {WebSocketSenderInterface} from "./web_socket_background";
function  any_to_string(any_object: any): string {
        if (typeof any_object === 'string') {
            return any_object;
        } else if (typeof any_object === 'object') {
            // 如果是对象，检查是否有自定义的错误属性
            return JSON.stringify(any_object, null, 2);
        } else {
           return any_object;
        }
}
export class WebSocketWorkerFront implements WebSocketSenderInterface {
    private web_socket_worker:SharedWorker;
    private callbackRegistry: { [callbackId: string]: (data: any) => void };
    constructor() {
        this.web_socket_worker=new SharedWorker('/script/web_socket_worker.ts');
        this.web_socket_worker.port.start();
        this.callbackRegistry = {};  // 存储回调函数的对象
    }

    // 注册回调函数并传递给全局注册器 (直接使用可序列化的 matchingFunction)
    registerCallbackWithFunction(matchingFunction: (data: any) => any, callback: (data: any) => void): void {
        this.callbackRegistry[callback.toString()] = callback;  // 在页面注册器中保存 callback

        // 传递回调 ID 和匹配函数给全局注册器
        this.web_socket_worker.port.postMessage({
            action: 'registerListenerWithFunction',
            callbackId: callback.toString(),
            matchingFunction: matchingFunction  // 假设 matchingFunction 可序列化
        });
    }

    // 注册回调函数并通过 key 来创建 matchingFunction
    registerCallbackWithKey(key: string, callback: (data: any) => void): void {
        this.callbackRegistry[callback.toString()] = callback;  // 在页面注册器中保存 callback

        // 传递回调 ID 和 key 给全局注册器
        this.web_socket_worker.port.postMessage({
            action: 'registerListenerWithKey',
            callbackId: callback.toString(),
            key: key  // 通过 key 创建 matchingFunction
        });
    }

    // 处理来自全局注册器的消息
    private handleWorkerMessage(callbackId: string, data: any): void {
        const callback = this.callbackRegistry[callbackId];  // 根据 callbackId 获取回调函数
        if (callback) {
            callback(data);  // 调用实际的回调函数
        } else {
            console.warn(`No callback found for callbackId: ${callbackId}`);
        }
    }

    // 移除回调函数
    removeCallback(callbackId: string): void {
        delete this.callbackRegistry[callbackId];
    }

    sendMessage(message: string): void {
                // 获取 token
        // 发送请求信息给 SharedWorker
        this.web_socket_worker.port.postMessage({
            action: "sendMessage",
            content: {message},
            axios_src: "https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"
        });
    }

    connect(): void {
        const token = localStorage.getItem("access_token");
        this.web_socket_worker.port.postMessage({
            action: "connect",
            content: {token}
        });
       this.web_socket_worker.port.onmessage = (event: MessageEvent) => {
            const { action, content } = event.data;
            if (action === "connected") {
                this.on_connected(content);
            }
            if (action === "disconnected") {
                this.on_disconnected(content);
            }
            if (action === "send_error") {
                this.on_send_error(content);
            }
            if (action === "socket_error") {
                this.on_socket_error(content);
            }
            if (action === 'notifyListener') {
                this.handleWorkerMessage(content.callbackId, content.data);
            }
        };
    }

    on_send_error(error: any): void {
        console.error("Error no message send.",any_to_string(error))
    }

    on_connected(event: any): void {
        console.error("web socket connected.",any_to_string(event))
    }

    on_disconnected(event: any): void {
        console.error("web socket closed.",any_to_string(event))
    }

    on_socket_error(error: any): void {
        console.error("web socket error.",any_to_string(error))
    }
}


