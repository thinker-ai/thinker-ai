import {RequestMessage, RequestSenderInterface} from "./request_sender_background";

export class RequestSenderWorkerFront implements RequestSenderInterface{
    request_sender_worker:SharedWorker;
    constructor() {
        this.request_sender_worker=new SharedWorker('/js/request_sender_worker.js');
        this.request_sender_worker.port.start();
    }
    makeRequest(message:RequestMessage): void
    {
        // 获取 token
        const token = localStorage.getItem("access_token");
        // 发送请求信息给 SharedWorker
        // 将 URLSearchParams 转换为键值对对象
        const paramsObject = message.params ? Object.fromEntries(message.params.entries()) : {};
        this.request_sender_worker.port.postMessage({
            action: "makeRequest",
            request: {
                method:message.method,
                url:message.url,
                paramsObject:paramsObject,
                body:message.body,
                token:message.token,
                content_type:message.content_type
            },
            token: token
        });
        // 处理来自 SharedWorker 的响应
        this.request_sender_worker.port.onmessage = (event: MessageEvent) => {
            const { action, response_data, error } = event.data;
            if (action === "on_response_ok" && message.on_response_ok) {
                message.on_response_ok(response_data)
            }
            else if (action === "on_response_error" && message.on_response_error) {
                message.on_response_error(error);
            }
            else if (action === "request_sender_worker_started") {
                console.info("Request sender worker started.");
            }
        };
    }
}


