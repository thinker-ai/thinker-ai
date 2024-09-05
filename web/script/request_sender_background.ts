// 导入 axios 类型
import { AxiosInstance, AxiosError,AxiosRequestConfig } from 'axios';
declare const axios: any;

function loadAxios(axios_src: string, token: string | null = null): Promise<AxiosInstance> {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = axios_src;  // 动态设置 axios 的路径
        script.onload = function () {
            const axiosInstance = axios.create();

            // 设置请求拦截器
            axiosInstance.interceptors.request.use(async function (config:any) {
                // 检查是否需要使用 token
                if (token && config.useToken) {
                    config.headers.Authorization = 'Bearer ' + token;
                }
                return config;
            }, function (error: AxiosError) {
                return Promise.reject(error);
            });

            resolve(axiosInstance); // 返回设置了拦截器的 axios 实例
        };
        script.onerror = function () {
            reject(new Error('Failed to load axios'));
        };
        document.head.appendChild(script);
    });
}

// 定义请求配置的接口
export interface RequestSenderInterface {
    makeRequest(
        method: 'get' | 'post',
        url: string,
        params?: object,
        useToken?: boolean,
        on_response_ok?:(response_data: any) => void,
        on_response_error?:(error_status: number|string) => void
    ): void;
}
// 定义 BaseRequestSender 类
export class RequestSender implements RequestSenderInterface {
    private axios_src: string;
    private token: string | null;
    protected constructor(axios_src: string, token: string | null = null) {
        this.axios_src = axios_src;
        this.token = token;
    }
    public makeRequest(method: 'get' | 'post',url: string,params?: object, useToken?: boolean,
        on_response_ok?:(response_data: any) => void,
        on_response_error?:(error_status: number|string) => void): void {
        loadAxios(this.axios_src, this.token).then((axiosInstance) => {
            // 创建 AxiosRequestConfig 配置对象，并初始化 headers 为空对象
            const config: AxiosRequestConfig = {
                method,
                url,
                params,
            };
            // 如果需要 token，则在请求头中添加 Authorization
            if (useToken && this.token) {
                config.headers = {
                    Authorization: `Bearer ${this.token}`
                };
            }
            axiosInstance(config).then(response => {
                if(on_response_ok){
                    on_response_ok(response.data);
                }
            }).catch((error) => {
                const error_status = error.response?.status || 'Unknown';
                if(on_response_error){
                    on_response_error(error_status);
                }
            });
        }).catch((error) => {
            console.error('Failed to load axios:', error);
        });
    }
}
export  class RequestSenderBackground extends RequestSender {
    private send_response: (response?: any) => void;

    constructor(axios_src: string, token: string | null = null, send_response: (response?: any) => void) {
        super(axios_src, token);
        this.send_response = send_response;
    }
    public makeRequest(method: 'get' | 'post',url: string,params?: object, useToken?: boolean): void {
        super.makeRequest(method,url,params,useToken,this.on_response_ok,this.on_response_error)
    }
    public on_response_ok(response_data: any) {
        this.send_response({ action: 'response_data', response_data });
    }

    public on_response_error(error_status: number|string) {
        this.send_response({ action: 'error', error: 'HTTP error! status: ' + error_status });
    }
}
