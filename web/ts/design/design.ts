// design.ts
import {query} from "../common.js";

export let processes: any[] = []; // 动态获取的流程数据

// 使用 query 方法请求数据
export function fetchProcesses(): void {
    query(
        '/design/solutions',
        (response_data) => {
            processes = response_data; // 将返回的数据赋值给 processes
            loadProcessList(); // 加载列表
            if (processes.length > 0) {
                loadContent(processes[0].id); // 默认加载第一个流程的内容
                document.querySelector(`#process-table-body tr[data-id="${processes[0].id}"]`)?.classList.add('selected');
            }
        },
        (error) => {
            console.error("Failed to fetch processes:", error);
        }
    );
}

// 初始化列表
export function loadProcessList(): void {
    const tbody = document.getElementById('process-table-body');
    if (!tbody) return;
    tbody.innerHTML = ''; // 清空现有内容
    processes.forEach(process => {
        const row = document.createElement('tr');
        row.dataset.id = process.id;
        row.innerHTML = `
            <td>${process.id}</td>
            <td>${process.name}</td>
            <td>${process.done ? "是" : "否"}</td>
            <td><span class="delete-btn" onclick="deleteProcess(${process.id}, event)">删除</span></td>
        `;
    row.onclick = (e) => {
        const target = e.target as HTMLElement | null; // 确保 e.target 的类型
        if (target && !target.classList.contains('delete-btn')) { // 检查 target 是否为 null
            selectRow(row);
            loadContent(process.id);
        }
    };
        tbody.appendChild(row);
    });
}

// 选中行高亮
export function selectRow(selectedRow: HTMLTableRowElement): void {
    const rows = document.querySelectorAll('#process-table-body tr');
    rows.forEach(row => row.classList.remove('selected')); // 移除所有行的选中状态
    selectedRow.classList.add('selected'); // 添加选中状态
}

// 加载内容到 iframe
export function loadContent(id: number): void {
    const contentFrame = document.getElementById('content-frame') as HTMLIFrameElement;
    if (!contentFrame) return;

    // 设置 iframe 的 src
    contentFrame.src = `/design/solution?id=${id}`;

    // 获取所有侧边栏按钮
    const sidebarButtons = document.querySelectorAll('.menu-item');
    sidebarButtons.forEach((button, index) => {
        // 提取原始 URL（去掉已有的 ?id 参数）
        const originalPage = button.getAttribute('onclick')?.split(',')[2]?.replace(/'/g, '').split('?')[0].trim();
        if (originalPage) {
            // 设置新的 URL 带上当前的 id
            button.setAttribute('onclick', `showContent('content-frame', 'menu-item', '${originalPage}?id=${id}', this)`);
        }

        // 恢复默认按钮的 active 状态（第一个按钮）
        if (index === 0) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

// 删除流程
export function deleteProcess(id: number, event: MouseEvent): void {
    event.stopPropagation(); // 阻止事件冒泡到行点击事件
    const index = processes.findIndex(p => p.id === id);
    if (index !== -1) {
        processes.splice(index, 1);
        loadProcessList();
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    fetchProcesses(); // 动态获取数据并初始化列表
});