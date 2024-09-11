import { registerCallbackWithKey, send_http } from "../../common";
import { initialize_floating_panel_if_extension_not_install } from "../floating-panel";
import {RequestMessage} from "../../request_sender_background";

interface ResponseData {
    name: string;
    description: string;
    solution_tree: TreeNode[];
}

interface TreeNode {
    name: string;
    description: string;
    children?: TreeNode[];
}

function submitProblem(): void {
    const titleElement = document.getElementById('problem-title') as HTMLInputElement;
    const descriptionElement = document.getElementById('problem-description') as HTMLTextAreaElement;
    if (titleElement.value === "" || descriptionElement.value === "") {
        alert("请输入您的问题!");
        return;
    }
    const data={
        name:titleElement.value,
        description:descriptionElement.value,
        is_root:true,
    }
    const request_message:RequestMessage={
            method:'post',
            url:'/design/one/solution/generate_state_machine_def',
            params:undefined,
            body:data
    }
    send_http(request_message);
}
(window as any).submitProblem = submitProblem;
function updateContent(content: string): void {
    const detailContainer = document.getElementById('detail-container') as HTMLElement;
    detailContainer.style.display = 'block';
    detailContainer.innerHTML = `<div id="detail-content">${content}</div>`;
}
(window as any).updateContent = updateContent;
function showData(): void {
    const request_message:RequestMessage={
            method:'get',
            url:'/design/one/solution/current',
    }
    send_http(request_message)
}
(window as any).showData = showData;
function showProblem(data: { name: string; description: string }): void {
    const problemTitle = document.getElementById('problem-title') as HTMLInputElement;
    problemTitle.value = data.name;

    const problemDescription = document.getElementById('problem-description') as HTMLTextAreaElement;
    problemDescription.value = data.description;
}
(window as any).showProblem = showProblem;
function showSolution(data: ResponseData): void {
    const solutionTree = document.getElementById('solution-tree') as HTMLElement;
    solutionTree.innerHTML = '';

    const rootNode = document.createElement('div');
    rootNode.className = 'tree-node';

    const rootNodeText = document.createElement('span');
    rootNodeText.className = 'node-text';
    rootNodeText.textContent = data.name;

    rootNodeText.onclick = () => {
        updateContent(data.description);
    };

    rootNode.appendChild(rootNodeText);
    solutionTree.appendChild(rootNode);

    function createTreeNode(nodeData: TreeNode): HTMLElement {
        const treeNode = document.createElement('div');
        treeNode.className = 'tree-node';

        if (nodeData.children && nodeData.children.length > 0) {
            const toggleSymbol = document.createElement('span');
            toggleSymbol.className = 'toggle-symbol';
            toggleSymbol.textContent = '-';
            toggleSymbol.onclick = () => {
                toggleNode(toggleSymbol);
            };
            treeNode.appendChild(toggleSymbol);

            const nodeText = document.createElement('span');
            nodeText.className = 'node-text';
            nodeText.textContent = nodeData.name;
            nodeText.onclick = () => {
                updateContent(nodeData.description);
            };
            treeNode.appendChild(nodeText);

            const childContainer = document.createElement('div');
            childContainer.className = 'child-nodes';
            childContainer.style.display = 'block';

            nodeData.children.forEach(child => {
                const childNode = createTreeNode(child);
                childContainer.appendChild(childNode);
            });

            treeNode.appendChild(childContainer);
        } else {
            const leafSymbol = document.createElement('span');
            leafSymbol.className = 'leaf-symbol';
            leafSymbol.textContent = '•';
            treeNode.appendChild(leafSymbol);

            const nodeText = document.createElement('span');
            nodeText.className = 'node-text';
            nodeText.textContent = nodeData.name;
            nodeText.onclick = () => {
                updateContent(nodeData.description);
            };
            treeNode.appendChild(nodeText);
        }

        return treeNode;
    }

    data.solution_tree.forEach(item => {
        const treeNode = createTreeNode(item);
        rootNode.appendChild(treeNode);
    });

    rootNodeText.click();
}
(window as any).showSolution = showSolution;
function toggleNode(symbol: HTMLElement): void {
    const node = symbol.parentElement;
    const childNodes = node?.querySelector('.child-nodes') as HTMLElement;

    if (childNodes) {
        if (childNodes.style.display === "none" || childNodes.style.display === "") {
            childNodes.style.display = "block";
            symbol.textContent = "-";
        } else {
            childNodes.style.display = "none";
            symbol.textContent = "+";
        }
    }
}
document.addEventListener('DOMContentLoaded', function() {
    initialize_floating_panel_if_extension_not_install(document.getElementById('content') as HTMLElement);
});
window.addEventListener('load', showData);
registerCallbackWithKey('command', ai_executor);
function ai_executor(data: any): void {
    console.info(data);
}
