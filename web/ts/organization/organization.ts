import {get_authorization, query, registerCallbackWithKey, send_http} from "../common.js";
import {RequestMessage} from "../request_sender_background";

interface ResponseData {
    name: string;
    description: string;
    organization_tree: TreeNode[];
}

interface TreeNode {
    name: string;
    description: string;
    children?: TreeNode[];
}

function updateContent(content: string): void {
    const detailContainer = document.getElementById('detail-container') as HTMLElement;
    detailContainer.style.display = 'block';
    detailContainer.innerHTML = `<div id="detail-content">${content}</div>`;
}
(window as any).updateContent = updateContent;

function showData(): void {
    query(
        '/organization',
        (response_data) => {
            showOrganization(response_data)
        },
        (error) => {
            console.error("response error:", error);
        }
    );
}
(window as any).showData = showData;
function showProblem(data: { name: string; description: string }): void {
    const problemTitle = document.getElementById('problem-title') as HTMLInputElement;
    problemTitle.value = data.name;

    const problemDescription = document.getElementById('problem-description') as HTMLTextAreaElement;
    problemDescription.value = data.description;
}
(window as any).showProblem = showProblem;
function showOrganization(data: ResponseData): void {
    const organizationTree = document.getElementById('organization-tree') as HTMLElement;
    organizationTree.innerHTML = '';

    const rootNode = document.createElement('div');
    rootNode.className = 'tree-node';

    const rootNodeText = document.createElement('span');
    rootNodeText.className = 'node-text';
    rootNodeText.textContent = data.name;

    rootNodeText.onclick = () => {
        updateContent(data.description);
    };

    rootNode.appendChild(rootNodeText);
    organizationTree.appendChild(rootNode);

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

    data.organization_tree.forEach(item => {
        const treeNode = createTreeNode(item);
        rootNode.appendChild(treeNode);
    });

    rootNodeText.click();
}
(window as any).showOrganization = showOrganization;
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
window.addEventListener('load', showData);
