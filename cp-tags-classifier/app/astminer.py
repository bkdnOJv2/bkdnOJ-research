from tree_sitter import Language, Parser, Node
import json

def content_leaf_node(node: Node):
    node_text = node.text.decode("utf-8")
    node_type = node.type
    if ' ' in node_text:
        return node_type
    if node_type.endswith('identifier') or \
        node_type == 'system_lib_string' or \
        node_type == 'primitive_type' or \
        node_type == 'number_literal':
        if len(node_text) > 15:
            return node_text[:15]
        return node_text
    return node_type


def extract_tokens(parser: Parser, src: str):
    tree = parser.parse(bytes(src, 'utf-8'))
    tokens = []   

    def dfs(node: Node):
        if node.children:
            for child in node.children:
                dfs(child)
        else:
            tokens.append(content_leaf_node(node))
    dfs(tree.root_node)
    
    return tokens

def extract_ast_path(parser: Parser, src: str, max_length=8):
    tree = parser.parse(bytes(src, "utf-8"))
    all_path = []
    cur_path = []

    with open('./model/vocab.json', 'r') as f:
        vocab_node = json.load(f)

    def dfs(node):
        if node.children:
            for child in node.children:
                if len(cur_path) > 1 and len(cur_path) <= max_length:
                    all_path.append(''.join(cur_path))
                cur_path.clear()
                dfs(child)
                cur_path.append('↑' + str(vocab_node.get(node.type, 0)))
        else:
            cur_path.append(str(vocab_node.get(node.type, 0)))

    dfs(tree.root_node)
    return all_path

def extract_ast_path2(parser: Parser, src: str, max_length=8):
    tree = parser.parse(bytes(src, "utf-8"))
    all_path = []
    cur_path = []


    def dfs(node):
        if node.children:
            for child in node.children:
                if len(cur_path) > 1 and len(cur_path) <= max_length:
                    all_path.append(''.join(cur_path))
                cur_path.clear()
                dfs(child)
                cur_path.append('↑' + node.type)
        else:
            cur_path.append(node.type)

    dfs(tree.root_node)
    return all_path

