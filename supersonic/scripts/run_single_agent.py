import supersonic.agent as agent

if __name__ == "__main__":
    sonic = agent.BaseAgent()
    sonic.train(1000, device='/gpu:1')
    sonic.save_weights('model_zoo/firstTryweights')

