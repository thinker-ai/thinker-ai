import {RequestSenderInterface} from "./request_sender_background";

export class RequestSenderWorkerFront implements RequestSenderInterface{
    request_sender_worker:SharedWorker;
    constructor() {
        this.request_sender_worker=new SharedWorker('/js/request_sender_worker.js');
        this.request_sender_worker.port.start();
    }
    makeRequest(method: string,url: string, params?: URLSearchParams,body?: any, useToken?: boolean,
                on_response_ok?:(response_data: any) => void,
                on_response_error?:(error_status: number|string) => void) {
        // 获取 token
        const token = localStorage.getItem("access_token");
        // 设置 useToken 标志
        useToken = token ? useToken : false;
        // 发送请求信息给 SharedWorker
        this.request_sender_worker.port.postMessage({
            action: "makeRequest",
            request: { method, url, params, body, useToken },
            token: token
        });
        // 处理来自 SharedWorker 的响应
        this.request_sender_worker.port.onmessage = (event: MessageEvent) => {
            const { action, response_data } = event.data;
            if (action === "response_ok" && on_response_ok) {
                on_response_ok(response_data)
            }
            if (action === "response_error" && on_response_error) {
                on_response_error(response_data);
            }
            if (action === "request_sender_worker_started") {
                console.info("Request sender worker started.");
            }
        };
    }
}


