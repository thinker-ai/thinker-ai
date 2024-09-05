import {RequestSenderInterface} from "./request_sender_background";

export class RequestSenderWorkerFront implements RequestSenderInterface{
    request_sender_worker:SharedWorker;
    constructor() {
        this.request_sender_worker=new SharedWorker('/script/request_sender_worker.ts');
        this.request_sender_worker.port.start();
    }
    makeRequest(method: string,url: string, params?: any, useToken?: boolean,
                on_response_ok?:(response_data: any) => void,
                on_response_error?:(error_status: number|string) => void) {
        // 获取 token
        const token = localStorage.getItem("access_token");
        // 设置 useToken 标志
        useToken = token ? useToken : false;
        // 发送请求信息给 SharedWorker
        this.request_sender_worker.port.postMessage({
            action: "makeRequest",
            request: { method, url, params, useToken },
            token: token,
            axios_src: "https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"
        });
        // 处理来自 SharedWorker 的响应
        this.request_sender_worker.port.onmessage = (event: MessageEvent) => {
            const { action, response_data } = event.data;
            if (action === "response_data" && on_response_ok) {
                on_response_ok(response_data)
            }
            if (action === "error" && on_response_error) {
                on_response_error(response_data);
            }
            if (action === "request_sender_worker_started") {
                console.error("Request sender worker started.");
            }
        };
    }
}


