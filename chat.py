from engines import MlpEngine, SvmEngine

def main():
    mlp = MlpEngine()
    svm = SvmEngine()

    engine = "mlp"
    print("Bot spreman. Promjena engine: /engine mlp ili /engine svm")

    while True:
        msg = input("Ti: ")
        if msg.lower() in ["kraj", "exit"]:
            break

        if msg.startswith("/engine"):
            engine = msg.split()[1]
            print(f"Koristim engine: {engine}")
            continue

        if engine == "svm":
            reply, _ = svm.reply(msg)
        else:
            reply, _ = mlp.reply(msg)

        print("Bot:", reply)

if __name__ == "__main__":
    main()
