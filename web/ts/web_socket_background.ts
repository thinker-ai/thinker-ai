
export interface WebSocketSenderInterface {
    on_send_error(error: any): void;
    connect(token:string): void;
    send_message(message:string): void;
    on_connected(event: any): void;
    on_disconnected(event: any): void;
    on_socket_error(error: any): void;
}
export abstract class AbstractWebSocketSender implements WebSocketSenderInterface{
    url='ws://localhost:8000/ws'
    socket!:WebSocket|null;
    reconnectInterval = 10000; // 30 second
    protected constructor(url?: string) {
        this.url = url || this.url;  // 如果 url 未传递，使用默认值
    }
    // WebSocket 连接函数
    public connect(token:string) {
        if (!token) {
            console.log('No token found.');
            return;
        }

        // 如果已经存在一个打开的 WebSocket 连接，不再重新连接
        if (this.socket && this.socket.readyState !== WebSocket.CLOSED && this.socket.readyState !== WebSocket.CLOSING) {
            console.log('A connection is already open. No need to reconnect.');
            return;
        }

        // 使用 Sec-WebSocket-Protocol 头传递 token
            // 将 token 作为 URL 参数传递
        const urlWithToken = `${this.url}?token=${token}`;
        this.socket = new WebSocket(urlWithToken);

        this.socket.onopen = () => {
            console.log('Connected to server');
            this.on_connected('Connected to server')
            this.reconnectInterval = 1000; // Reset the interval on successful connection

            // Start sending heartbeat messages every 10 seconds
            const heartbeatInterval = setInterval(() => {
                if (this.socket!.readyState === WebSocket.OPEN) {
                    this.socket!.send(JSON.stringify({type: 'heartbeat'}));
                }
            }, 10000);
            this.socket!.onclose = () => clearInterval(heartbeatInterval);
        };

        this.socket.onmessage = (event:MessageEvent) => {
            const data = JSON.parse(event.data);
            if (data.type !== 'heartbeat') {
                // 如果是在插件环境或普通网页中，使用其他方法，例如 chrome.runtime 或直接触发事件
                this.on_receive_message(data.message)
            }
        }

        this.socket.onclose = (event:CloseEvent) => {
            console.log('Connection closed', event);
            this.on_disconnected(event)
            if (!event.wasClean) {
                console.log('Connection closed due to network or server issues.');
                setTimeout(() => this.connect(token), this.reconnectInterval); // Attempt to reconnect after a delay
                this.reconnectInterval = Math.min(this.reconnectInterval * 2, 5000); // Max 5 seconds
            }
        };

        this.socket.onerror = (error:Event) => {
            console.log('WebSocket error', error);
            this.socket!.close(); // 关闭当前连接，触发 onclose 事件，进而重新连接
            this.on_socket_error(error)
        };
    }
// 使用 WebSocket 发送消息
    public send_message(message:any) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({type: 'message', message}));
        } else {
            console.log('WebSocket is not open. Message not sent.');
        }

    }
    public disconnect(): void{
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            try{
                this.socket.close();
                this.on_disconnected("closed")
            }catch(e){
                console.log('WebSocket close error');
            }finally {
                this.socket=null;
            }
        } else {
            console.log('WebSocket is not open');
        }
    }
    protected abstract on_receive_message(message: any): void;
    abstract on_send_error(error: any): void;
    abstract on_connected(event: any): void;
    abstract on_disconnected(event: any): void;
    abstract on_socket_error(error: any): void;
}

interface Listener {
    matchingFunction: (data: any) => any;
    callbackId: string;
}

export class WebSocketSenderBackgroundWithCallback extends AbstractWebSocketSender {
    private readonly send_response: (response?: any) => void;
    private listeners: Listener[] = [];

    public constructor(send_response: (response?: any) => void, url?: string) {
        super(url);
        this.send_response = send_response;
    }

    // 注册带有可序列化的匹配函数的监听器
    public register_function_listener(matchingFunction: (data: any) => any, callbackId: string): void {
        this.listeners.push({ matchingFunction, callbackId });
    }

    // 通过 key 注册监听器
    public register_key_listener(key: string, callbackId: string): void {
        const matchingFunction = this.create_key_matcher(key);
        this.listeners.push({ matchingFunction, callbackId });
    }

    // 定义 createKeyMatcher 函数，生成一个匹配函数
    private create_key_matcher(key: string) {
        return (data: Record<string, any> | null): any => {
            try {
                if (data && typeof data === 'object' && key in data) {
                    return data[key]; // 匹配成功，返回该 key 对应的值
                }
            } catch (e) {
                console.error('Error while matching key:', e);
            }
            return null; // 匹配失败
        };
    }
    protected on_receive_message(message: any): void {
        this.listeners.forEach(listener => {
            const result = listener.matchingFunction(message);
            if (result) {
                // 通过 port 发送匹配结果和 callbackId 给页面
                this.send_response({ action: 'callback', callbackId: listener.callbackId, data: result });
            }
        });
    }
    // 发送错误时触发
    public on_send_error(error: any): void {
        this.send_response({ action: 'send_error', error });
    }

    // 连接成功时触发
    public on_connected(event: any): void {
        this.send_response({ action: 'connected', data: event });
    }

    // 断开连接时触发
    public on_disconnected(event: any): void {
        this.send_response({ action: 'disconnected', data: event });
    }

    // WebSocket 错误时触发
    public on_socket_error(error: any): void {
        this.send_response({ action: 'socket_error', error });
    }
}
