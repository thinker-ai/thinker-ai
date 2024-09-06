import { RequestSenderWorkerFront } from "./request_sender_front";
import { WebSocketWorkerFront } from "./web_socket_front";
export function showContent(frame_id:string,menu_class_name:string,page:string,element:Element):void {
    // Update the iframe's src attribute to load the corresponding page
    const html_element = window.document.getElementById(frame_id)
    if (html_element && 'src' in html_element) {
        html_element.src = page;
    }
    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName(menu_class_name);
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}
const request_sender_worker_client = new RequestSenderWorkerFront()
export function makeRequest(method:'get'|'post', url:string, params={},useToken=true,
                            on_response_ok?:(response_data: any) => void,
                            on_response_error?:(error_status: number|string) => void) {
    request_sender_worker_client.makeRequest(method, url, params, useToken,on_response_ok,on_response_error);
}
const web_socket_worker_client = new WebSocketWorkerFront()
export function send_websocket_message(message:string):void {
    web_socket_worker_client.sendMessage(message);
}

export function registerCallbackWithKey(key:string,callback: (data: any) => void):void {
    web_socket_worker_client.registerCallbackWithKey(key,callback)
}
export function registerCallbackWithFunction(matchingFunction: (data: any) => any,callback: (data: any) => void):void {
    web_socket_worker_client.registerCallbackWithFunction(matchingFunction,callback)
}


export function resolve_authorization_result(): Promise<{ user_id: string; access_token: string } | null> {
    return new Promise((resolve, reject) => {
        // 监听来自 content script 的消息
        const messageHandler = (event: MessageEvent) => {
            if (event.data && event.data.action === 'authorizationResult') {
                window.removeEventListener('message', messageHandler);  // 确保只处理一次消息
                resolve(event.data.response);  // 解析消息数据并返回
            }
        };

        window.addEventListener('message', messageHandler);

        // 发送消息给 content script，请求获取 authorizationResult 信息
        window.postMessage({ action: 'getAuthorization' }, '*');

        // 设置超时，如果超过一定时间没有接收到消息，则 reject
        const timeout = setTimeout(() => {
            window.removeEventListener('message', messageHandler);  // 超时后移除事件监听器
            reject(new Error('Authorization result not received within timeout period'));
        }, 5000); // 5秒超时，可以根据实际需求调整

    });
}
export function run_after_plugin_checked(onInstalled?:() => void, onNotInstalled?:() => void) {
    let pluginChecked = false;

    // 事件处理器
    function messageHandler(event:MessageEvent) {
        if (event.source === window && event.data.action === 'thinkerAI') {
            pluginChecked = true;  // 标记为已检查插件
            if (event.data.value === true && onInstalled) {
                onInstalled();  // 插件已安装，执行回调
            }
            window.removeEventListener('message', messageHandler);  // 移除事件监听器
        }
    }

    // 监听消息
    window.addEventListener('message', messageHandler);

    // 发送消息以检查插件是否安装
    window.postMessage({ action: 'isThinkerAIInstall' }, '*');

    // 延时检查，给插件响应时间
    setTimeout(() => {
        if (!pluginChecked && onNotInstalled) {
            onNotInstalled();  // 插件未安装，执行回调
        }
    }, 500); // 设置合理的延迟时间
}
run_after_plugin_checked(
    undefined,
    web_socket_worker_client.connect
);
