import subprocess

# Execute the code in PoT method / during Execution Stage
def process_code(code, code_file="code_exec/tmp0"):
    try:
        code_pieces = []
        while "```python" in code:
            st_idx = code.index("```python") + 10
            end_idx = code.index("```", st_idx)
            code_pieces.append(code[st_idx:end_idx].strip())
            code = code[end_idx+3:].strip()
        if len(code_pieces) == 0:
            return False, "No code found"
        else:
            code = "\n\n".join(code_pieces)
            f = open(f"{code_file}.py", "w")
            f.write(code)
            f.close()
            result = subprocess.run(
                ['python', f'{code_file}.py'], capture_output=True, check=False, text=True, timeout=1)
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif code_file in m:
                        st = m.index('"/') + 1
                        ed = m.index(f'/{code_file}.py') + 1
                        clr = m[st:ed]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
            else:
                output = result.stdout
                # print("Code Run successfully!")
                return True, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout detected in running subprocess"
    except Exception as e:
        return False, "Unknown error in running subprocess"
