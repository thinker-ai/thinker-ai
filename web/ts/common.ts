import { RequestSenderWorkerFront } from "./request_sender_front";
import { WebSocketWorkerFront } from "./web_socket_front";
import {RequestMessage} from "./request_sender_background";
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
// 将 showContent 函数挂载到 window 对象上
(window as any).showContent = showContent;

const request_sender_worker_client = new RequestSenderWorkerFront()
export function send_http(message:RequestMessage):void
{
    request_sender_worker_client.send_http(message);
}
const web_socket_worker_client = new WebSocketWorkerFront()
export function send_websocket(message:string):void {
    web_socket_worker_client.send_message(message);
}

export function registerCallbackWithKey(key:string,callback: (data: any) => void):void {
    web_socket_worker_client.register_callback_with_key(key,callback)
}
export function registerCallbackWithFunction(matchingFunction: (data: any) => any,callback: (data: any) => void):void {
    web_socket_worker_client.register_callback_with_function(matchingFunction,callback)
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
    () => web_socket_worker_client.connect()
);
